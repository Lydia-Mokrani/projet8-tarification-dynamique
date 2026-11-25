from types import SimpleNamespace

import numpy as np

from ecom_agents.agents.pricing import PricingAgent
from ecom_agents.config import Config


def _make_world(n_products=2, n_weeks=50, price_hist=None, comp_price=None, units_hist=None, cost=None, p_ref=None):
    if price_hist is None:
        price_hist = np.full((n_products, n_weeks), 20.0)
    if comp_price is None:
        comp_price = np.full((n_products, n_weeks), 20.0)
    if units_hist is None:
        units_hist = np.full((n_products, n_weeks), 50.0)
    if cost is None:
        cost = np.full(n_products, 10.0)
    if p_ref is None:
        p_ref = np.full(n_products, 20.0)
    return SimpleNamespace(
        cost=cost,
        p_ref=p_ref,
        capacity=None,
        price_hist=price_hist,
        comp_price=comp_price,
        units_hist=units_hist,
    )


def test_floor_ceiled_survives_orchestrator_rounding():
    cfg = Config(n_products=1, n_weeks=50, test_weeks=10)
    cost = np.array([9.094])  # 9.094 * 1.10 = 10.0034 -> round(..., 2) alone gives 10.00 < floor
    ag = PricingAgent(cfg).fit(_make_world(n_products=1, cost=cost, p_ref=np.array([10.0])), np.full((1, 50), 0.5))
    ag.set_segments(np.array([True]))

    prices, info = ag.run_batch(
        p_prev=np.array([10.0]),
        pc_prev=np.array([9.0]),
        rep=np.array([0.5]),
        q0=np.array([50.0]),
        sensitive=np.array([True]),
    )
    raw_floor = cost * (1 + cfg.min_margin)
    final_prices = np.round(np.maximum(prices, raw_floor), 2)
    assert info[0]["lo"] == 10.01
    assert prices[0] == 10.01
    assert np.all(final_prices >= raw_floor)


def test_empty_segment_no_nan():
    cfg = Config(n_products=3, n_weeks=50, test_weeks=10, shrink=0.5)
    ag = PricingAgent(cfg).fit(_make_world(n_products=3), np.full((3, 50), 0.5))

    for mask in (np.array([True, True, True]), np.array([False, False, False])):
        ag.set_segments(mask)
        assert np.all(np.isfinite(ag.eps))
        assert np.isfinite(ag.seg_eps[True])
        assert np.isfinite(ag.seg_eps[False])


def test_fit_zero_and_nan_guards():
    cfg = Config(n_products=2, n_weeks=50, test_weeks=10)
    rng = np.random.default_rng(0)
    p = np.full((2, 50), 20.0)
    p[0, :35] = 20.0 * rng.uniform(0.85, 1.15, 35)
    p[0, 35:38] = 0.0
    p[0, 38:40] = np.nan
    p[1, :20] = 0.0  # only 20 valid weeks (< 30)

    q = np.full((2, 50), 50.0)
    q[0, :35] = 50.0 * (p[0, :35] / 20.0) ** (-2.0)
    q[0, 35:40] = 50.0
    q[1, :10] = np.nan
    q[1, 10:20] = 0.0

    c = np.full((2, 50), 20.0)
    r = np.full((2, 50), 0.5)
    r[0, 39] = np.nan

    ag = PricingAgent(cfg).fit(_make_world(n_products=2, price_hist=p, comp_price=c, units_hist=q), r)
    assert np.all(np.isfinite(ag.eps_hat))
    assert -4.0 <= ag.eps_hat[0] <= -0.3
    assert ag.eps_hat[1] == -1.5  # fallback when < 30 valid rows


def test_overrides_eps_and_clipping():
    cfg = Config(n_products=2, n_weeks=50, test_weeks=10)
    ag = PricingAgent(cfg).fit(_make_world(n_products=2), np.full((2, 50), 0.5))
    ag.set_segments(np.array([True, False]))

    _, info_dict = ag.run_batch(
        np.array([20.0, 20.0]),
        np.array([20.0, 20.0]),
        np.array([0.5, 0.5]),
        np.array([50.0, 50.0]),
        np.array([True, False]),
        overrides_eps={0: -10.0, 1: 0.5},
    )
    assert info_dict[0]["eps"] == -4.0
    assert info_dict[1]["eps"] == -0.3

    _, info_seq = ag.run_batch(
        np.array([20.0, 20.0]),
        np.array([20.0, 20.0]),
        np.array([0.5, 0.5]),
        np.array([50.0, 50.0]),
        np.array([True, False]),
        overrides_eps=[-2.2, np.nan],
    )
    assert info_seq[0]["eps"] == -2.2
    assert info_seq[1]["eps"] == ag.eps[1]


def test_hi_lt_lo_reason_and_info_keys():
    cfg = Config(n_products=2, n_weeks=50, test_weeks=10)
    ag = PricingAgent(cfg).fit(
        _make_world(n_products=2, cost=np.array([9.094, 10.0]), p_ref=np.array([10.0, 20.0])),
        np.full((2, 50), 0.5),
    )
    ag.set_segments(np.array([True, True]))

    prices, info = ag.run_batch(
        p_prev=np.array([10.0, 20.0]),
        pc_prev=np.array([9.0, 10.0]),
        rep=np.array([0.5, 0.5]),
        q0=np.array([50.0, 50.0]),
        sensitive=np.array([True, True]),
    )
    assert prices[0] == 10.01
    assert info[0]["reason"] == "borne basse (plancher_marge) prioritaire sur plafond_concurrent"
    assert prices[1] == 19.0
    assert info[1]["reason"] == "borne basse (variation_max) prioritaire sur plafond_concurrent"
    for item in info:
        assert set(item.keys()) == {"eps", "p_star_non_contraint", "lo", "hi", "reason"}
