"""Ablation study for DemandAgent LSTM on validation weeks [t_val, t0) only.

Tests incremental changes one at a time over 5 seeds without touching existing modules
and without peeking at test weeks [t0, n_weeks).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecom_agents.agents.demand import harmonics
from ecom_agents.agents.reputation import ReputationAgent
from ecom_agents.config import Config
from ecom_agents.data import generate_world


def harmonics_k4(w) -> np.ndarray:
    w = np.atleast_1d(w).astype(float)
    return np.stack([f(2 * np.pi * k * w / 52) for k in (1, 2, 3, 4) for f in (np.sin, np.cos)], axis=1)


class AblationLSTMNet(nn.Module):
    def __init__(self, d_in: int, d_extra: int, hidden: int = 32, use_std: bool = False, use_skip: bool = False):
        super().__init__()
        self.use_std = use_std
        self.use_skip = use_skip
        self.register_buffer("x_mean", torch.zeros(1, 1, d_in))
        self.register_buffer("x_std", torch.ones(1, 1, d_in))
        self.register_buffer("e_mean", torch.zeros(1, d_extra))
        self.register_buffer("e_std", torch.ones(1, d_extra))
        self.lstm = nn.LSTM(d_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + d_extra, 32), nn.ReLU(), nn.Linear(32, 1))
        self.skip = nn.Linear(d_in + d_extra, 1) if use_skip else None

    def set_norm(self, X: np.ndarray, E: np.ndarray):
        xm = X.mean(axis=(0, 1), keepdims=True)
        xs = np.where(X.std(axis=(0, 1), keepdims=True) > 1e-6, X.std(axis=(0, 1), keepdims=True), 1.0)
        em = E.mean(axis=0, keepdims=True)
        es = np.where(E.std(axis=0, keepdims=True) > 1e-6, E.std(axis=0, keepdims=True), 1.0)
        self.x_mean.copy_(torch.tensor(xm, dtype=torch.float32))
        self.x_std.copy_(torch.tensor(xs, dtype=torch.float32))
        self.e_mean.copy_(torch.tensor(em, dtype=torch.float32))
        self.e_std.copy_(torch.tensor(es, dtype=torch.float32))

    def forward(self, x, e):
        if self.use_std:
            x = (x - self.x_mean) / self.x_std
            e = (e - self.e_mean) / self.e_std
        _, (hn, _) = self.lstm(x)
        out = self.head(torch.cat([hn[-1], e], dim=1)).squeeze(1)
        if self.skip is not None:
            out = out + self.skip(torch.cat([x[:, -1, :], e], dim=1)).squeeze(1)
        return out


def build_static_meta(world) -> np.ndarray:
    cats = sorted(world.products["category"].unique())
    cat_oh = np.column_stack([(world.products["category"].to_numpy() == c).astype(float) for c in cats])
    lp_ref = np.log(world.p_ref)
    lp_ref = (lp_ref - lp_ref.mean()) / (lp_ref.std() + 1e-6)
    margin_rate = 1.0 - world.cost / world.p_ref
    margin_rate = (margin_rate - margin_rate.mean()) / (margin_rate.std() + 1e-6)
    return np.column_stack([cat_oh, lp_ref, margin_rate])


def build_dataset(world, rep_hat, stat_end: int, w_start: int, w_end: int, opts: dict, static_meta: np.ndarray):
    mq = np.log(world.units_hist[:, :stat_end]).mean(1)
    mp = np.log(world.price_hist[:, :stat_end]).mean(1)
    mc = np.log(world.comp_price[:, :stat_end]).mean(1)
    F = np.stack([
        np.log(world.units_hist) - mq[:, None],
        np.log(world.price_hist) - mp[:, None],
        np.log(world.comp_price) - mc[:, None],
        rep_hat - 0.5,
    ], axis=2)
    h_fn = harmonics_k4 if opts["harmonics_k4"] else harmonics
    n, win = F.shape[0], world.cfg.window
    Xs, Es, ys = [], [], []
    for w in range(w_start, w_end):
        X = F[:, w - win:w, :]
        dlogp = np.log(world.price_hist[:, w]) - np.log(world.price_hist[:, w - 1])
        parts = [dlogp[:, None], np.repeat(h_fn(w), n, axis=0)]
        if opts["log_price_level"]:
            parts.append((np.log(world.price_hist[:, w]) - mp)[:, None])
        if opts["static_meta"]:
            parts.append(static_meta)
        E = np.concatenate(parts, axis=1)
        Xs.append(X); Es.append(E); ys.append(F[:, w, 0])
    return np.concatenate(Xs), np.concatenate(Es), np.concatenate(ys), mq, mp, mc, F


def train_model(X, E, y, seed: int, opts: dict, X_val=None, E_val=None, y_val=None, epochs_override: int | None = None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    Et = torch.tensor(E, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    net = AblationLSTMNet(X.shape[2], E.shape[1], hidden=32, use_std=opts["use_std"], use_skip=opts["use_skip"])
    if opts["use_std"]:
        net.set_norm(X, E)

    if opts["adamw_clip_es"]:
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    else:
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        sched = None

    loss_fn = nn.SmoothL1Loss(beta=0.1) if opts["smooth_l1"] else nn.MSELoss()
    max_epochs = epochs_override if epochs_override is not None else 60
    batch_size = 128
    n = len(Xt)

    use_es = opts["adamw_clip_es"] and (X_val is not None)
    if use_es:
        Xv = torch.tensor(X_val, dtype=torch.float32)
        Ev = torch.tensor(E_val, dtype=torch.float32)
        yv = torch.tensor(y_val, dtype=torch.float32)
        best_val = float("inf")
        best_state = copy.deepcopy(net.state_dict())
        best_epoch = max_epochs
        no_imp = 0

    for ep in range(1, max_epochs + 1):
        net.train()
        perm = torch.randperm(n)
        for k in range(0, n, batch_size):
            idx = perm[k:k + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(Xt[idx], Et[idx]), yt[idx])
            loss.backward()
            if opts["adamw_clip_es"]:
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        if use_es:
            net.eval()
            with torch.no_grad():
                vloss = float(loss_fn(net(Xv, Ev), yv).item())
            sched.step(vloss)
            if vloss < best_val - 1e-5:
                best_val = vloss
                best_state = copy.deepcopy(net.state_dict())
                best_epoch = ep
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= 10:
                    break

    if use_es:
        net.load_state_dict(best_state)
        net.eval()
        return net, best_epoch
    net.eval()
    return net, max_epochs


def eval_on_val(world, rep_hat, t_es: int, t_val: int, t0: int, seed: int, opts: dict, static_meta: np.ndarray) -> float:
    win = world.cfg.window
    if opts["adamw_clip_es"]:
        X_tr, E_tr, y_tr, _, _, _, _ = build_dataset(world, rep_hat, t_es, win, t_es, opts, static_meta)
        X_es, E_es, y_es, _, _, _, _ = build_dataset(world, rep_hat, t_es, t_es, t_val, opts, static_meta)
        _, best_epoch = train_model(X_tr, E_tr, y_tr, seed, opts, X_val=X_es, E_val=E_es, y_val=y_es)
        X_full, E_full, y_full, mq, mp, _, F = build_dataset(world, rep_hat, t_val, win, t_val, opts, static_meta)
        net, _ = train_model(X_full, E_full, y_full, seed, opts, epochs_override=best_epoch)
    else:
        X_full, E_full, y_full, mq, mp, _, F = build_dataset(world, rep_hat, t_val, win, t_val, opts, static_meta)
        net, _ = train_model(X_full, E_full, y_full, seed, opts)

    h_fn = harmonics_k4 if opts["harmonics_k4"] else harmonics
    n = F.shape[0]
    mapes = []
    net.eval()
    with torch.no_grad():
        for w in range(t_val, t0):
            X = F[:, w - win:w, :]
            dlogp = np.log(world.price_hist[:, w]) - np.log(world.price_hist[:, w - 1])
            parts = [dlogp[:, None], np.repeat(h_fn(w), n, axis=0)]
            if opts["log_price_level"]:
                parts.append((np.log(world.price_hist[:, w]) - mp)[:, None])
            if opts["static_meta"]:
                parts.append(static_meta)
            E = np.concatenate(parts, axis=1)
            pred_c = net(torch.tensor(X, dtype=torch.float32), torch.tensor(E, dtype=torch.float32)).numpy()
            pred = np.exp(pred_c + mq)
            actual = world.units_hist[:, w]
            mapes.append(np.abs(pred - actual) / actual)
    return float(np.mean(mapes))


def eval_baselines_on_val(world, rep_hat, t_val: int, t0: int) -> tuple[float, float]:
    base_opts = {"use_std": False, "use_skip": False, "adamw_clip_es": False, "smooth_l1": False, "log_price_level": False, "harmonics_k4": False, "static_meta": False}
    static_meta = build_static_meta(world)
    X_tr, E_tr, y_tr, mq, _, _, F = build_dataset(world, rep_hat, t_val, world.cfg.window, t_val, base_opts, static_meta)
    flat = lambda x, e: np.concatenate([x.reshape(len(x), -1), e], axis=1)
    ridge = Ridge(alpha=1.0).fit(flat(X_tr, E_tr), y_tr)
    ridge_mapes, naive_mapes = [], []
    n, win = F.shape[0], world.cfg.window
    for w in range(t_val, t0):
        X = F[:, w - win:w, :]
        dlogp = np.log(world.price_hist[:, w]) - np.log(world.price_hist[:, w - 1])
        E = np.concatenate([dlogp[:, None], np.repeat(harmonics(w), n, axis=0)], axis=1)
        actual = world.units_hist[:, w]
        ridge_pred = np.exp(ridge.predict(flat(X, E)) + mq)
        ridge_mapes.append(np.abs(ridge_pred - actual) / actual)
        naive_mapes.append(np.abs(world.units_hist[:, w - 1] - actual) / actual)
    return float(np.mean(ridge_mapes)), float(np.mean(naive_mapes))


def main():
    seeds = [1, 2, 3, 4, 5]
    worlds = []
    for s in seeds:
        cfg = Config(seed=s)
        t0 = cfg.n_weeks - cfg.test_weeks
        t_val = t0 - 12
        t_es = t_val - 12
        w = generate_world(cfg)
        rep = ReputationAgent(cfg, backend="light").fit(w.reviews, t_val)
        rep_hat = rep.build_series(w.reviews, cfg.n_products, cfg.n_weeks)
        worlds.append((s, w, rep_hat, build_static_meta(w), t_es, t_val, t0))

    ridge_scores, naive_scores = [], []
    for _, w, rep_hat, _, _, t_val, t0 in worlds:
        rm, nm = eval_baselines_on_val(w, rep_hat, t_val, t0)
        ridge_scores.append(rm); naive_scores.append(nm)

    base_opts = {"use_std": False, "use_skip": False, "adamw_clip_es": False, "smooth_l1": False, "log_price_level": False, "harmonics_k4": False, "static_meta": False}
    steps = [
        ("0_baseline", {}),
        ("1_std_skip", {"use_std": True, "use_skip": True}),
        ("2_adamw_clip_es", {"adamw_clip_es": True}),
        ("3_smooth_l1", {"smooth_l1": True}),
        ("4_log_price_level", {"log_price_level": True}),
        ("5_harmonics_k4", {"harmonics_k4": True}),
        ("6_static_meta", {"static_meta": True}),
    ]

    current_opts = dict(base_opts)
    best_mean = float("inf")
    ablation_rows = []
    per_seed = {}

    print(f"Validation window: weeks [{worlds[0][5]}, {worlds[0][6]}) | 5 seeds: {seeds}", flush=True)
    print(f"Ridge (unchanged) : {np.mean(ridge_scores):.5f} +/- {np.std(ridge_scores):.5f}", flush=True)
    print(f"Naive             : {np.mean(naive_scores):.5f} +/- {np.std(naive_scores):.5f}\n", flush=True)
    print(f"{'Step':<22} | {'Val MAPE mean +/- std':<25} | {'Delta vs best':<14} | {'Decision'}", flush=True)
    print("-" * 78, flush=True)

    for name, delta_opts in steps:
        trial_opts = {**current_opts, **delta_opts}
        scores = [eval_on_val(w, rep_hat, t_es, t_val, t0, s, trial_opts, sm) for s, w, rep_hat, sm, t_es, t_val, t0 in worlds]
        per_seed[name] = scores
        m, sd = float(np.mean(scores)), float(np.std(scores))
        if name == "0_baseline":
            best_mean = m
            current_opts = trial_opts
            delta_str, decision = "-", "BASELINE"
        elif m < best_mean - 1e-6:
            delta_str = f"{m - best_mean:+.5f}"
            best_mean = m
            current_opts = trial_opts
            decision = "KEEP"
        else:
            delta_str = f"{m - best_mean:+.5f}"
            decision = "DROP"
        ablation_rows.append({"step": name, "mean": m, "std": sd, "per_seed": scores, "decision": decision})
        print(f"{name:<22} | {m:.5f} +/- {sd:.5f}       | {delta_str:<14} | {decision}", flush=True)

    step2_opts = {**base_opts, "adamw_clip_es": True}
    diag_rows = []
    print("\nDiagnostic on top of 2_adamw_clip_es:", flush=True)
    for dname, extra in (("step2_plus_std_only", {"use_std": True}), ("step2_plus_skip_only", {"use_skip": True})):
        scores = [eval_on_val(w, rep_hat, t_es, t_val, t0, s, {**step2_opts, **extra}, sm) for s, w, rep_hat, sm, t_es, t_val, t0 in worlds]
        m, sd = float(np.mean(scores)), float(np.std(scores))
        delta = m - float(np.mean(per_seed["2_adamw_clip_es"]))
        decision = "KEEP" if delta < -1e-6 else "DROP"
        per_seed[dname] = scores
        diag_rows.append({"step": dname, "mean": m, "std": sd, "delta_vs_step2": delta, "per_seed": scores, "decision": decision})
        print(f"  {dname:<22} | {m:.5f} +/- {sd:.5f} | delta vs step2: {delta:+.5f} | {decision}", flush=True)

    paired = [b - s2 for b, s2 in zip(per_seed["0_baseline"], per_seed["2_adamw_clip_es"])]
    n_improved = sum(d > 0 for d in paired)
    print(f"\nPaired diff (0_baseline - 2_adamw_clip_es) per seed:", flush=True)
    for s, b, s2, d in zip(seeds, per_seed["0_baseline"], per_seed["2_adamw_clip_es"], paired):
        print(f"  seed={s}: baseline={b:.5f}, step2={s2:.5f}, diff={d:+.5f} ({'improved' if d > 0 else 'worse'})", flush=True)
    print(f"Improved seeds: {n_improved}/{len(seeds)} | mean diff: {np.mean(paired):+.5f}", flush=True)

    out_path = Path(__file__).resolve().parent / "lstm_ablation_results.json"
    out_data = {
        "validation_weeks": [int(worlds[0][5]), int(worlds[0][6])],
        "seeds": seeds,
        "ridge": {"mean": float(np.mean(ridge_scores)), "std": float(np.std(ridge_scores)), "per_seed": ridge_scores},
        "naive": {"mean": float(np.mean(naive_scores)), "std": float(np.std(naive_scores)), "per_seed": naive_scores},
        "ablation": ablation_rows,
        "diagnostics_on_step2": diag_rows,
        "paired_baseline_minus_step2": {"per_seed": paired, "improved_count": int(n_improved), "mean_diff": float(np.mean(paired))},
    }
    out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
