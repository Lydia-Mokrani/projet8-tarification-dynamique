from types import SimpleNamespace

import numpy as np

from ecom_agents.agents.demand import DemandAgent
from ecom_agents.config import Config
from ecom_agents.nn_models import LSTMNet, predict_lstm, train_lstm


def _make_world(n_products=4, n_weeks=50, seed=0):
    rng = np.random.default_rng(seed)
    p = 20.0 * rng.uniform(0.9, 1.1, (n_products, n_weeks))
    c = 20.0 * rng.uniform(0.9, 1.1, (n_products, n_weeks))
    q = 50.0 * (p / 20.0) ** (-1.8) * rng.normal(1.0, 0.05, (n_products, n_weeks))
    q = np.maximum(q, 1.0)
    return SimpleNamespace(units_hist=q, price_hist=p, comp_price=c)


def test_features_no_crash_on_zero_and_nan():
    cfg = Config(n_products=3, n_weeks=30, test_weeks=6, window=6, lstm_epochs=5)
    w = _make_world(n_products=3, n_weeks=30)
    rep = np.full((3, 30), 0.5)
    w.units_hist[0, :3] = 0.0
    w.units_hist[1, 3:5] = np.nan
    w.price_hist[2, 0] = -1.0
    w.comp_price[0, 4] = np.nan
    rep[1, 2] = np.nan

    ag = DemandAgent(cfg, backend="light").fit(w, rep)
    F = ag.features(w.units_hist, w.price_hist, w.comp_price, rep)
    assert F.shape == (3, 30, 4)
    assert np.all(np.isfinite(F))
    assert np.all(np.isfinite(ag.mq))
    assert np.all(np.isfinite(ag.mp))
    assert np.all(np.isfinite(ag.mc))


def test_no_future_leakage_from_test_weeks():
    for backend in ("light", "torch"):
        cfg = Config(seed=7, n_products=4, n_weeks=50, test_weeks=10, window=8, lstm_epochs=6)
        t0 = cfg.n_weeks - cfg.test_weeks
        w1 = _make_world(n_products=4, n_weeks=50, seed=7)
        rep1 = np.full((4, 50), 0.5)

        w2 = _make_world(n_products=4, n_weeks=50, seed=7)
        rep2 = np.full((4, 50), 0.5)
        w2.units_hist[:, t0:] = 99999.0
        w2.price_hist[:, t0:] = 500.0
        w2.comp_price[:, t0:] = 1.0
        rep2[:, t0:] = 0.0

        ag1 = DemandAgent(cfg, backend=backend).fit(w1, rep1)
        ag2 = DemandAgent(cfg, backend=backend).fit(w2, rep2)

        np.testing.assert_allclose(ag1.mq, ag2.mq)
        np.testing.assert_allclose(ag1.mp, ag2.mp)
        np.testing.assert_allclose(ag1.mc, ag2.mc)

        st1 = SimpleNamespace(units=w1.units_hist.copy(), price=w1.price_hist.copy(), comp=w1.comp_price.copy(), rep_hat=rep1.copy())
        st2 = SimpleNamespace(units=w2.units_hist.copy(), price=w2.price_hist.copy(), comp=w2.comp_price.copy(), rep_hat=rep2.copy())
        st1.units[:, t0:], st1.price[:, t0:] = np.nan, np.nan
        st2.units[:, t0:], st2.price[:, t0:] = np.nan, np.nan

        p1 = ag1.run_batch(st1, t0)
        p2 = ag2.run_batch(st2, t0)
        assert p1.shape == (4,)
        assert np.all(np.isfinite(p1)) and np.all(p1 > 0)
        np.testing.assert_allclose(p1, p2, rtol=1e-6, atol=1e-6)


def test_predict_shape_and_forecast_eval():
    for backend in ("light", "torch"):
        cfg = Config(seed=11, n_products=5, n_weeks=60, test_weeks=12, window=10, lstm_epochs=8)
        w = _make_world(n_products=5, n_weeks=60, seed=11)
        rep = np.full((5, 60), 0.5)
        ag = DemandAgent(cfg, backend=backend).fit(w, rep)
        st = SimpleNamespace(units=w.units_hist.copy(), price=w.price_hist.copy(), comp=w.comp_price.copy(), rep_hat=rep)
        preds = ag.run_batch(st, cfg.n_weeks - cfg.test_weeks)
        assert preds.shape == (5,)
        assert np.all(np.isfinite(preds)) and np.all(preds > 0)
        ev = ag.forecast_eval(w, rep)
        assert set(ev.keys()) == {f"MAPE_{ag.model_name}", "MAPE_ridge", "MAPE_naive"}
        assert all(np.isfinite(v) and v >= 0 for v in ev.values())


def test_train_lstm_backward_compatible_and_eval_mode():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 6, 4)).astype(np.float32)
    E = rng.normal(size=(32, 7)).astype(np.float32)
    y = rng.normal(size=(32,)).astype(np.float32)
    net = train_lstm(X, E, y, seed=0, epochs=2)
    assert isinstance(net, LSTMNet)
    assert not net.training
    net.train()
    out = predict_lstm(net, X[:4], E[:4])
    assert out.shape == (4,)
    assert np.all(np.isfinite(out))
    assert not net.training
