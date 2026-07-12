#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
# Avoid BLAS/threadpool hangs in constrained notebook or CI environments.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_funds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    dates = pd.date_range("2019-09-30", "2026-01-31", freq="Q")
    cats = ["EC", "STIV", "DBT", "DE", "EP", "ABS-MBS", "RA", "DIR", "DFE", "LON"]
    rows = []
    long_rows = []
    summary_rows = []
    for i in range(args.n_funds):
        series_id = f"S{i:05d}"
        cl = rng.integers(0, 4)
        base_alpha = np.ones(len(cats)) * 0.4
        if cl == 0:
            base_alpha[cats.index("EC")] = 8
        elif cl == 1:
            base_alpha[cats.index("DBT")] = 6
            base_alpha[cats.index("STIV")] = 3
        elif cl == 2:
            base_alpha[cats.index("STIV")] = 8
        else:
            base_alpha[cats.index("EP")] = 3
            base_alpha[cats.index("EC")] = 3
            base_alpha[cats.index("DIR")] = 2
        is_etf = int(cl in [0, 1] and rng.random() < 0.4)
        is_index = int(is_etf and rng.random() < 0.6)
        derivative_allowed = int(cl == 3 or rng.random() < 0.25)
        short_allowed = int(cl == 3 and rng.random() < 0.7)
        leverage = int(cl == 3 and rng.random() < 0.5)
        drp = rng.poisson(0.5 if cl != 3 else 1.2)
        net_assets = rng.lognormal(18, 1)
        theta_flow = rng.normal(0, 1)
        prev_w = rng.dirichlet(base_alpha)
        for t, dt in enumerate(dates):
            shock = rng.normal(0, 0.02, len(cats))
            w = 0.85 * prev_w + 0.15 * rng.dirichlet(np.maximum(base_alpha + shock, 0.05))
            w = np.maximum(w, 0)
            w = w / w.sum()
            prev_w = w
            vix = 16 + 8 * np.sin(t / 5) + rng.normal(0, 4)
            hy = 3.5 + 0.5 * np.cos(t / 4) + rng.normal(0, 0.2)
            rate10 = 2 + 1.5 * np.sin(t / 7) + rng.normal(0, 0.1)
            derivative_pct = max(0, rng.normal(0.03 if derivative_allowed else 0.005, 0.025))
            short_pct = max(0, rng.normal(0.02 if short_allowed else 0.002, 0.015))
            restricted_pct = max(0, rng.normal(0.015, 0.01))
            us_pct = min(1, max(0.3, rng.normal(0.8 if cl != 3 else 0.65, 0.1)))
            non_us_pct = max(0, 1 - us_pct + rng.normal(0, 0.02))
            cash_not_reported = net_assets * max(0, rng.normal(0.03 if cl == 2 else 0.01, 0.01))
            flow = net_assets * (0.005 * theta_flow - 0.002 * vix/20 + rng.normal(0, 0.02))
            sales = max(flow, 0) + abs(rng.normal(0, 0.01 * net_assets))
            redempt = max(-flow, 0) + abs(rng.normal(0, 0.01 * net_assets))
            accession = f"A{i:05d}_{t:03d}"
            row = {
                "series_id": series_id,
                "accession_number": accession,
                "report_date": dt,
                "filing_date": dt + pd.Timedelta(days=60),
                "net_assets": net_assets * rng.lognormal(0, 0.05),
                "total_assets": net_assets * rng.lognormal(0.02, 0.05),
                "cash_not_reported": cash_not_reported,
                "sales_flow_3m": sales,
                "redemption_flow_3m": redempt,
                "net_flow_3m": sales - redempt,
                "n_holdings": int(rng.lognormal(4.5, 0.5)),
                "total_currency_value": net_assets,
                "total_percentage": 1.0,
                "long_pct": 1.0 + derivative_pct,
                "short_pct": short_pct,
                "derivative_pct": derivative_pct,
                "restricted_pct": restricted_pct,
                "us_pct": us_pct,
                "non_us_pct": non_us_pct,
                "IS_ETF": is_etf,
                "IS_INDEX": is_index,
                "IS_FUND_OF_FUND": int(rng.random() < 0.12),
                "IS_NON_DIVERSIFIED": int(cl == 3),
                "IS_SEC_LENDING_AUTHORIZED": int(rng.random() < 0.35),
                "DID_LEND_SECURITIES": int(rng.random() < 0.2),
                "HAS_LINE_OF_CREDIT": int(rng.random() < 0.15),
                "HAS_INTERFUND_LENDING": int(rng.random() < 0.10),
                "HAS_SWING_PRICING": int(rng.random() < 0.12),
                "MANAGEMENT_FEE": max(0, rng.normal(0.004 + 0.002 * (cl == 3), 0.001)),
                "NET_OPERATING_EXPENSES": max(0, rng.normal(0.006 + 0.003 * (cl == 3), 0.0015)),
                "n_advisers": 1 + int(rng.random() < 0.15),
                "n_custodians": 1 + int(rng.random() < 0.10),
                "n_pricing_services": 1 + int(rng.random() < 0.25),
                "n_brokers": 2 + rng.poisson(2),
                "adv_total_drp_records": drp,
                "allows_or_discusses_derivatives": derivative_allowed,
                "shorting": short_allowed,
                "leverage": leverage,
                "liquidity_or_redemption_constraint": int(cl == 2 or rng.random() < 0.1),
                "discretionary_management": 1,
                "wrap_fee": int(rng.random() < 0.1),
                "performance_fee": int(cl == 3 and rng.random() < 0.35),
                "esg": int(rng.random() < 0.1),
                "crypto": int(rng.random() < 0.03),
                "legal_disciplinary": int(drp > 0),
                "custody": int(rng.random() < 0.3),
                "vix": vix,
                "vxv": vix + rng.normal(1, 2),
                "hy_oas": hy,
                "rate_10y": rate10,
                "rate_2y": rate10 + rng.normal(0, 0.3),
                "SPY_63d_ret": rng.normal(0.02, 0.06),
                "QQQ_63d_ret": rng.normal(0.025, 0.08),
                "IWM_63d_ret": rng.normal(0.018, 0.08),
                "TLT_63d_ret": rng.normal(0.005, 0.05),
                "EEM_63d_ret": rng.normal(0.012, 0.08),
                "GLD_63d_ret": rng.normal(0.01, 0.06),
                "SPY_63d_vol": rng.uniform(0.10, 0.30),
                "TLT_63d_vol": rng.uniform(0.06, 0.20),
                "GLD_63d_vol": rng.uniform(0.10, 0.28),
                "SPY_drawdown_63d": -abs(rng.normal(0.05, 0.05)),
                "mkt_rf": rng.normal(0.0003, 0.01),
                "smb": rng.normal(0, 0.005),
                "hml": rng.normal(0, 0.005),
                "rmw": rng.normal(0, 0.004),
                "cma": rng.normal(0, 0.004),
                "mom": rng.normal(0, 0.006),
            }
            rows.append(row)
            summary_rows.append({k: row[k] for k in ["series_id", "report_date", "n_holdings", "total_currency_value", "total_percentage", "long_pct", "short_pct", "derivative_pct", "restricted_pct", "us_pct", "non_us_pct"]})
            for cat, pct in zip(cats, w):
                long_rows.append({"series_id": series_id, "accession_number": accession, "report_date": dt, "asset_cat": cat, "payoff_profile": "Long", "holding_count": int(max(1, pct * row["n_holdings"])), "currency_value_sum": pct * net_assets, "percentage_sum": pct})
    pd.DataFrame(rows).to_csv(out / "nport_choice_with_market_adviser.csv", index=False)
    pd.DataFrame(long_rows).to_csv(out / "nport_asset_class_exposure_long.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "nport_fund_choice_summary.csv", index=False)
    print(f"Synthetic data written to {out}")


if __name__ == "__main__":
    main()
