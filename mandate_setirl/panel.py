from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import coerce_datetime, percent_to_decimal, present_cols, safe_divide, smart_read_table, smart_write_table, robust_zscore, write_json


@dataclass
class PanelData:
    panel: pd.DataFrame
    action_cols: list[str]
    asset_action_cols: list[str]
    overlay_action_cols: list[str]
    state_cols: list[str]
    entity_col: str
    date_col: str


def _file_path(data_dir: Path, filename: str | None) -> Path | None:
    if not filename:
        return None
    p = Path(filename)
    if p.is_absolute():
        return p
    return data_dir / p


def load_raw_inputs(cfg: dict[str, Any], data_dir_override: str | None = None) -> dict[str, pd.DataFrame | None]:
    data_dir = Path(data_dir_override or cfg["data_dir"])
    files = cfg.get("files", {})
    out: dict[str, pd.DataFrame | None] = {}
    main_path = _file_path(data_dir, files.get("main_panel"))
    if main_path is None:
        raise ValueError("files.main_panel is required")
    out["main_panel"] = smart_read_table(main_path)
    for key in ["asset_exposure_long", "fund_choice_summary", "adv_contract_constraints"]:
        path = _file_path(data_dir, files.get(key))
        if path is not None and path.exists():
            out[key] = smart_read_table(path)
        else:
            out[key] = None
    return out


def _normalise_key_series(s: pd.Series) -> pd.Series:
    """Normalize filing/accession/CRD keys for robust CSV merges."""
    x = s.astype(str).str.strip()
    x = x.str.replace(r"\.0$", "", regex=True)
    x = x.replace({"nan": "", "None": "", "NaT": ""})
    return x


def _binary_or_numeric(s: pd.Series) -> pd.Series:
    """Convert ADV keyword flags to numeric robustly.

    Real files may store 0/1, True/False, yes/no, or non-empty keyword text.
    Non-empty non-false text is treated as a soft positive keyword hit.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    x = s.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    true = x.isin(["1", "true", "t", "yes", "y", "present", "discussed", "allowed", "allow"])
    false = x.isin(["0", "false", "f", "no", "n", "nan", "none", "", "not present"])
    out[true] = 1.0
    out[false] = 0.0
    out[out.isna() & ~false & x.ne("") & x.ne("nan") & x.ne("none")] = 1.0
    return out



def _summary_base_col(col: str) -> str:
    for suffix in ("_neg_abs", "_pos"):
        if col.endswith(suffix):
            return col[: -len(suffix)]
    return col


def _clean_summary_action_series(s: pd.Series, col: str, cfg: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    """Clean overlay percentage action columns after unit conversion.

    Real N-PORT summary fields can contain economically impossible outliers from
    derivative notionals, sign conventions, or data glitches.  These values must
    not enter the action space unbounded, otherwise mandate bounds, candidate
    projection, and utility features become meaningless.  The defaults are fixed
    economic caps, not sample quantiles, to avoid time leakage.
    """
    clean_cfg = cfg.get("actions", {}).get("summary_action_cleaning", {}) or {}
    enabled = bool(clean_cfg.get("enabled", True))
    x0 = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x0.copy()
    stats: dict[str, Any] = {
        "column": col,
        "enabled": enabled,
        "n": int(len(x0)),
        "n_missing_before_fill": int(x0.isna().sum()),
        "raw_min": float(x0.min(skipna=True)) if x0.notna().any() else np.nan,
        "raw_p01": float(x0.quantile(0.01)) if x0.notna().any() else np.nan,
        "raw_p99": float(x0.quantile(0.99)) if x0.notna().any() else np.nan,
        "raw_max": float(x0.max(skipna=True)) if x0.notna().any() else np.nan,
    }
    if not enabled:
        return x.fillna(0.0), stats
    nonneg = set(clean_cfg.get("nonnegative_cols", []))
    lower_bounds = dict(clean_cfg.get("lower_bounds", {}) or {})
    upper_bounds = dict(clean_cfg.get("upper_bounds", {}) or {})
    base_col = _summary_base_col(col)
    lower = lower_bounds.get(col, lower_bounds.get(base_col, 0.0 if col in nonneg or base_col in nonneg else None))
    upper = upper_bounds.get(col, upper_bounds.get(base_col, None))
    if lower is not None:
        stats["n_lower_clipped"] = int((x < float(lower)).sum(skipna=True))
        x = x.clip(lower=float(lower))
    else:
        stats["n_lower_clipped"] = 0
    if upper is not None:
        stats["n_upper_clipped"] = int((x > float(upper)).sum(skipna=True))
        x = x.clip(upper=float(upper))
    else:
        stats["n_upper_clipped"] = 0
    denom = max(1, int(len(x0)))
    stats["lower_clip_rate"] = float(stats["n_lower_clipped"] / denom)
    stats["upper_clip_rate"] = float(stats["n_upper_clipped"] / denom)
    stats["total_clip_rate"] = float((stats["n_lower_clipped"] + stats["n_upper_clipped"]) / denom)
    stats["lower_bound"] = lower
    stats["upper_bound"] = upper
    stats["clean_min"] = float(x.min(skipna=True)) if x.notna().any() else np.nan
    stats["clean_p01"] = float(x.quantile(0.01)) if x.notna().any() else np.nan
    stats["clean_p99"] = float(x.quantile(0.99)) if x.notna().any() else np.nan
    stats["clean_max"] = float(x.max(skipna=True)) if x.notna().any() else np.nan
    return x.fillna(0.0), stats


def _prepare_summary_action_columns(main: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    """Convert and clean summary overlay actions.

    Besides the default clipped version, robustness configs can either drop rows
    with contaminated signed exposures or split signed columns into positive and
    negative-absolute components.
    """
    main = main.copy()
    actions_cfg = cfg.get("actions", {})
    clean_cfg = actions_cfg.get("summary_action_cleaning", {}) or {}
    pct_mode = actions_cfg.get("percent_cols_are_0_100", "auto")
    summary_cols = [c for c in actions_cfg.get("summary_action_cols", []) if c in main.columns]
    converted = {c: percent_to_decimal(main[c], pct_mode) for c in summary_cols}
    reports: list[dict[str, Any]] = []

    drop_cfg = clean_cfg.get("drop_rows", {}) or {}
    if bool(drop_cfg.get("enabled", False)) and converted:
        bad = pd.Series(False, index=main.index)
        lower_bounds = dict(drop_cfg.get("lower_bounds", {}) or {})
        upper_bounds = dict(drop_cfg.get("upper_bounds", {}) or {})
        columns = list(drop_cfg.get("columns") or sorted(set(lower_bounds) | set(upper_bounds)))
        for c in columns:
            if c not in converted:
                continue
            x = pd.to_numeric(converted[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if c in lower_bounds:
                bad = bad | (x < float(lower_bounds[c]))
            if c in upper_bounds:
                bad = bad | (x > float(upper_bounds[c]))
        n_before = int(len(main))
        n_drop = int(bad.sum())
        if n_drop:
            main = main.loc[~bad].copy()
            converted = {c: x.loc[main.index] for c, x in converted.items()}
        reports.append({
            "column": "__row_filter__",
            "enabled": True,
            "n": n_before,
            "n_rows_dropped": n_drop,
            "drop_rate": float(n_drop / max(n_before, 1)),
            "drop_columns": ",".join(columns),
        })

    overlay_cols: list[str] = []
    signed_split_cols = set(clean_cfg.get("signed_split_cols", []) or [])
    for c in summary_cols:
        if c not in converted:
            continue
        x = pd.to_numeric(converted[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if c in signed_split_cols:
            split_series = {
                f"{c}_pos": x.clip(lower=0.0),
                f"{c}_neg_abs": (-x).clip(lower=0.0),
            }
            for out_col, sx in split_series.items():
                cleaned, stats = _clean_summary_action_series(sx, out_col, cfg)
                stats["source_column"] = c
                stats["signed_split_part"] = out_col[len(c) + 1 :]
                main[out_col] = cleaned
                reports.append(stats)
                overlay_cols.append(out_col)
        else:
            cleaned, stats = _clean_summary_action_series(x, c, cfg)
            main[c] = cleaned
            reports.append(stats)
            overlay_cols.append(c)
    return main, overlay_cols, reports

def make_asset_action_pivot(
    long_df: pd.DataFrame | None,
    cfg: dict[str, Any],
    entity_col: str,
    date_col: str,
    accession_col: str | None,
    main_key_map: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Pivot asset-class exposure into action columns.

    When the long table lacks series and date keys, accession numbers recover
    them from the main N-PORT panel.
    """
    if long_df is None or long_df.empty:
        return None, []
    df = long_df.copy()

    # Recover keys from accession_number when the long table lacks them.
    if (entity_col not in df.columns or date_col not in df.columns) and accession_col and accession_col in df.columns and main_key_map is not None:
        needed = [accession_col, entity_col, date_col]
        keymap = main_key_map[[c for c in needed if c in main_key_map.columns]].copy()
        if set(needed).issubset(keymap.columns):
            keymap = keymap.dropna(subset=[accession_col]).drop_duplicates([accession_col, entity_col, date_col])
            df["__accession_key__"] = _normalise_key_series(df[accession_col])
            keymap["__accession_key__"] = _normalise_key_series(keymap[accession_col])
            # Require accession numbers to identify a unique fund-date row.
            amb = keymap.groupby("__accession_key__")[[entity_col, date_col]].nunique(dropna=True)
            ambiguous_keys = amb[(amb[entity_col] > 1) | (amb[date_col] > 1)].index.tolist()
            if ambiguous_keys and not cfg.get("actions", {}).get("allow_ambiguous_accession_asset_merge", False):
                sample = ambiguous_keys[:5]
                raise ValueError(
                    f"asset_exposure_long lacks {entity_col}/{date_col}, but accession_number is not unique in main_panel. "
                    f"Examples: {sample}. Pre-merge asset exposure with series_id/report_date or set "
                    "actions.allow_ambiguous_accession_asset_merge=true only if duplication is intended."
                )
            keymap = keymap.drop(columns=[accession_col])
            df = df.merge(keymap, on="__accession_key__", how="left", suffixes=("", "__main"))
            df = df.drop(columns=["__accession_key__"])
            for c in [entity_col, date_col]:
                alt = f"{c}__main"
                if alt in df.columns:
                    if c in df.columns:
                        df[c] = df[c].where(df[c].notna(), df[alt])
                        df = df.drop(columns=[alt])
                    else:
                        df = df.rename(columns={alt: c})

    if date_col not in df.columns:
        possible_dates = [c for c in ["report_date", "REPORT_DATE", "period_of_report", "quarter"] if c in df.columns]
        if possible_dates and possible_dates[0] != "quarter":
            df[date_col] = df[possible_dates[0]]
        else:
            raise ValueError(
                f"{date_col} missing from asset exposure long file and could not be recovered from {accession_col}. "
                "Provide accession_number in both main_panel and asset_exposure_long, or pre-merge series_id/report_date."
            )
    coerce_datetime(df, date_col)
    if entity_col not in df.columns:
        raise ValueError(
            f"{entity_col} missing from asset exposure long file and could not be recovered from {accession_col}. "
            "Provide accession_number in both main_panel and asset_exposure_long, or pre-merge series_id/report_date."
        )
    if "asset_cat" not in df.columns or "percentage_sum" not in df.columns:
        return None, []
    df = df.dropna(subset=[entity_col, date_col]).copy()
    df["asset_cat"] = df["asset_cat"].where(df["asset_cat"].notna(), "UNKNOWN").astype(str)
    df["percentage_sum"] = percent_to_decimal(df["percentage_sum"], cfg["actions"].get("percent_cols_are_0_100", "auto"))
    top = list(cfg["actions"].get("top_asset_cats", []))
    df["asset_cat_clean"] = np.where(df["asset_cat"].isin(top), df["asset_cat"], "OTHER")
    idx_cols = [entity_col, date_col]
    piv = (
        df.groupby(idx_cols + ["asset_cat_clean"], dropna=False)["percentage_sum"]
        .sum()
        .reset_index()
        .pivot_table(index=idx_cols, columns="asset_cat_clean", values="percentage_sum", aggfunc="sum", fill_value=0.0)
        .reset_index()
    )
    asset_cols = []
    for cat in top + ["OTHER"]:
        if cat in piv.columns:
            new_col = f"asset_{cat.replace('-', '_').replace('/', '_')}"
            piv[new_col] = pd.to_numeric(piv[cat], errors="coerce").fillna(0.0)
            asset_cols.append(new_col)
    piv = piv[idx_cols + asset_cols]
    if cfg["actions"].get("normalize_asset_weights", True) and asset_cols:
        arr = piv[asset_cols].clip(lower=0.0).to_numpy(dtype=float)
        denom = arr.sum(axis=1, keepdims=True)
        denom[denom <= 0] = 1.0
        piv[asset_cols] = arr / denom
    return piv, asset_cols

def _merge_summary_actions(main: pd.DataFrame, summary: pd.DataFrame | None, entity_col: str, date_col: str, accession_col: str | None = None) -> pd.DataFrame:
    if summary is None or summary.empty:
        return main
    summary = summary.copy()
    if date_col in summary.columns:
        coerce_datetime(summary, date_col)
    merge_cols = [entity_col, date_col]
    if all(c in summary.columns for c in merge_cols):
        add_cols = [c for c in summary.columns if c not in main.columns and c not in merge_cols]
        if add_cols:
            return main.merge(summary[merge_cols + add_cols].drop_duplicates(merge_cols), on=merge_cols, how="left")
        return main
    if accession_col and accession_col in main.columns and accession_col in summary.columns:
        add_cols = [c for c in summary.columns if c not in main.columns and c != accession_col]
        if add_cols:
            left = main.copy()
            right = summary[[accession_col] + add_cols].copy()
            left["__accession_key__"] = _normalise_key_series(left[accession_col])
            right["__accession_key__"] = _normalise_key_series(right[accession_col])
            right = right.drop(columns=[accession_col]).drop_duplicates("__accession_key__")
            out = left.merge(right, on="__accession_key__", how="left")
            return out.drop(columns=["__accession_key__"])
    return main


_CANONICAL_ALIASES: dict[str, list[str]] = {
    "allows_or_discusses_derivatives": ["allows_or_discusses_derivatives", "mentions_derivatives", "derivatives"],
    "shorting": ["shorting", "allows_or_discusses_shorting", "mentions_shorting"],
    "leverage": ["leverage", "allows_or_discusses_leverage", "mentions_leverage"],
    "liquidity_or_redemption_constraint": ["liquidity_or_redemption_constraint", "mentions_liquidity_or_redemption_constraint", "liquidity_redemption_constraint"],
    "discretionary_management": ["discretionary_management", "mentions_discretionary_management", "allows_or_discusses_discretionary_management"],
    "wrap_fee": ["wrap_fee", "mentions_wrap_fee"],
    "performance_fee": ["performance_fee", "mentions_performance_fee"],
    "esg": ["esg", "mentions_esg", "allows_or_discusses_esg"],
    "crypto": ["crypto", "mentions_crypto", "allows_or_discusses_crypto"],
    "legal_disciplinary": ["legal_disciplinary", "mentions_legal_disciplinary"],
    "custody": ["custody", "mentions_custody"],
    "adv_total_drp_records": ["adv_total_drp_records", "total_drp_records", "adviser_drp_total", "adv_drp_total", "adv_total_drp", "adv_disciplinary_total"],
    "ia_regulatory_total": ["ia_regulatory_total", "ia_regulatory", "ia_regulatory_count", "ia_regulatory_total_records"],
}


def _coalesce_to_canonical(df: pd.DataFrame, canonical: str, aliases: list[str]) -> None:
    """Coalesce aliases into a canonical numeric/flag column.

    The row-wise maximum preserves positive keyword flags and aggregates count
    aliases such as DRP records.
    """
    found = [c for c in aliases if c in df.columns]
    if not found:
        return
    vals = []
    for c in found:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            vals.append(pd.to_numeric(s, errors="coerce"))
        else:
            vals.append(_binary_or_numeric(s))
    mat = pd.concat(vals, axis=1)
    out = mat.max(axis=1, skipna=True)
    out = out.where(mat.notna().any(axis=1), np.nan)
    df[canonical] = out


def standardize_column_aliases(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    out = df.copy()
    aliases = dict(_CANONICAL_ALIASES)
    aliases.update((cfg or {}).get("column_aliases", {}) or {})
    for canonical, cols in aliases.items():
        _coalesce_to_canonical(out, canonical, list(dict.fromkeys([canonical] + list(cols))))

    # Normalize string-valued keyword flags before numeric conversion.
    flag_cols = [
        "allows_or_discusses_derivatives", "shorting", "leverage",
        "liquidity_or_redemption_constraint", "discretionary_management",
        "wrap_fee", "performance_fee", "esg", "crypto",
        "legal_disciplinary", "custody",
    ]
    for c in flag_cols:
        if c in out.columns:
            out[c] = _binary_or_numeric(out[c]).fillna(0.0)
    for c in ["adv_total_drp_records", "ia_regulatory_total"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _aggregate_adv_contract_constraints(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Standardize ADV contract labels without prematurely choosing a merge key.

    Real ADV mapping files can have several brochures per filing, CRD, or SEC
    number. The merge routine aggregates by the actual key being used, taking
    max over binary keyword labels.
    """
    df = standardize_column_aliases(df, cfg)
    label_cols = list(dict.fromkeys(cfg.get("adv_contract_constraints", {}).get("label_cols", []) or [
        "allows_or_discusses_derivatives", "shorting", "leverage",
        "liquidity_or_redemption_constraint", "discretionary_management", "wrap_fee",
        "performance_fee", "esg", "crypto", "legal_disciplinary", "custody",
    ]))
    label_cols = [c for c in label_cols if c in df.columns]
    key_candidates = [
        "filing_id", "adv_filing_id", "brochure_id", "crd_number", "sec_number", "sec_file_number",
        "primary_adv_filing_id", "primary_adviser_crd", "primary_adviser_file_num", "primary_sec_file_number",
    ]
    keep = [c for c in key_candidates + label_cols if c in df.columns]
    if not keep:
        return pd.DataFrame()
    out = df[keep].copy()
    for c in label_cols:
        out[c] = _binary_or_numeric(out[c]).fillna(0.0)
    return out


def merge_adv_contract_constraints(main: pd.DataFrame, adv: pd.DataFrame | None, cfg: dict[str, Any]) -> pd.DataFrame:
    if adv is None or adv.empty:
        return standardize_column_aliases(main, cfg)
    main = standardize_column_aliases(main, cfg)
    adv = _aggregate_adv_contract_constraints(adv, cfg)
    if adv.empty:
        return main
    merge_specs = cfg.get("adv_contract_constraints", {}).get("merge_keys") or [
        ["primary_adv_filing_id", "filing_id"], ["primary_adv_filing_id", "adv_filing_id"],
        ["primary_adviser_crd", "crd_number"], ["primary_adviser_file_num", "sec_file_number"],
        ["adv_filing_id", "filing_id"], ["adv_filing_id_asof", "filing_id"],
        ["filing_id", "filing_id"], ["adviser_filing_id", "filing_id"],
        ["adviser_crd_num", "crd_number"], ["crd_number", "crd_number"],
        ["sec_file_number", "sec_file_number"], ["sec_file_number", "sec_number"],
        ["adviser_file_num", "sec_file_number"],
    ]
    label_cols = [c for c in [
        "allows_or_discusses_derivatives", "shorting", "leverage",
        "liquidity_or_redemption_constraint", "discretionary_management", "wrap_fee",
        "performance_fee", "esg", "crypto", "legal_disciplinary", "custody",
    ] if c in adv.columns]
    best = None
    best_rate = -1.0
    best_spec = None
    for left_key, right_key in merge_specs:
        if left_key not in main.columns or right_key not in adv.columns:
            continue
        left = main[[left_key]].copy()
        right = adv[[right_key] + label_cols].copy()
        left["__merge_key__"] = _normalise_key_series(left[left_key])
        right["__merge_key__"] = _normalise_key_series(right[right_key])
        # Aggregate duplicate filings by the active merge key.
        if label_cols:
            for c in label_cols:
                right[c] = _binary_or_numeric(right[c]).fillna(0.0)
            right = right.drop(columns=[right_key]).groupby("__merge_key__", dropna=False)[label_cols].max().reset_index()
        else:
            right = right.drop(columns=[right_key]).drop_duplicates("__merge_key__")
        probe = left[["__merge_key__"]].merge(right, on="__merge_key__", how="left")
        rate = probe[label_cols].notna().any(axis=1).mean() if label_cols else 0.0
        if rate > best_rate:
            best_rate = float(rate)
            best = (left["__merge_key__"], right)
            best_spec = (left_key, right_key)
    acc_cfg = cfg.get("adv_contract_constraints", {})
    min_rate = float(acc_cfg.get("min_merge_rate", 0.0))
    fail_low = bool(acc_cfg.get("fail_on_low_merge_rate", False))
    if best is None or best_rate < min_rate:
        rate = best_rate if best_rate >= 0 else 0.0
        if fail_low:
            available_left = sorted(set(main.columns))
            available_right = sorted(set(adv.columns))
            tried = ", ".join([f"{l}->{r}" for l, r in merge_specs])
            raise ValueError(
                f"ADV contract merge failed or below threshold: best_rate={rate:.4f}, min_merge_rate={min_rate:.4f}. "
                f"Tried keys: {tried}. Check real main-panel keys such as primary_adv_filing_id or primary_adviser_crd. "
                f"Left columns sample: {available_left[:25]}; right columns sample: {available_right[:25]}"
            )
        main["adv_contract_merge_method"] = "none"
        main["adv_contract_merge_rate"] = rate
        return main
    left_keys, right = best
    tmp = main.copy()
    tmp["__merge_key__"] = left_keys.to_numpy()
    merged = tmp.merge(right, on="__merge_key__", how="left", suffixes=("", "_advcontract"))
    for c in label_cols:
        ac = f"{c}_advcontract"
        if ac in merged.columns:
            if c in merged.columns:
                left_val = _binary_or_numeric(merged[c])
                right_val = _binary_or_numeric(merged[ac])
                merged[c] = pd.concat([left_val, right_val], axis=1).max(axis=1, skipna=True)
            else:
                merged[c] = _binary_or_numeric(merged[ac])
            merged = merged.drop(columns=[ac])
    merged = merged.drop(columns=["__merge_key__"])
    merged["adv_contract_merge_method"] = f"{best_spec[0]}->{best_spec[1]}"
    merged["adv_contract_merge_rate"] = best_rate
    return standardize_column_aliases(merged, cfg)


def build_panel(cfg: dict[str, Any], data_dir_override: str | None = None, out_dir: str | None = None) -> PanelData:
    raw = load_raw_inputs(cfg, data_dir_override)
    main = raw["main_panel"].copy()
    main = merge_adv_contract_constraints(main, raw.get("adv_contract_constraints"), cfg)
    keys = cfg.get("keys", {})
    entity_col = keys.get("entity_col", "series_id")
    date_col = keys.get("date_col", "report_date")
    accession_col = keys.get("accession_col", "accession_number")
    if entity_col not in main.columns:
        raise ValueError(f"Main panel missing entity column: {entity_col}")
    if date_col not in main.columns:
        raise ValueError(f"Main panel missing date column: {date_col}")
    coerce_datetime(main, date_col)
    main = _merge_summary_actions(main, raw.get("fund_choice_summary"), entity_col, date_col, accession_col)
    main = standardize_column_aliases(main, cfg)

    asset_piv, asset_cols = make_asset_action_pivot(raw.get("asset_exposure_long"), cfg, entity_col, date_col, accession_col, main_key_map=main)
    if asset_piv is not None:
        main = main.merge(asset_piv, on=[entity_col, date_col], how="left")
    else:
        asset_cols = [c for c in main.columns if c.startswith("asset_")]
    main = standardize_column_aliases(main, cfg)

    main, overlay_cols, action_cleaning_reports = _prepare_summary_action_columns(main, cfg)

    cash_col = cfg["actions"].get("cash_col")
    net_assets_col = cfg["actions"].get("net_assets_col")
    if cash_col in main.columns and net_assets_col in main.columns:
        main["cash_proxy"] = safe_divide(main[cash_col], main[net_assets_col], default=0.0)
        main["cash_proxy"] = np.clip(main["cash_proxy"], -1.0, 1.0)
        overlay_cols.append("cash_proxy")

    n_hold = cfg["actions"].get("n_holdings_col")
    if n_hold in main.columns:
        main["n_holdings_scaled"] = robust_zscore(np.log1p(pd.to_numeric(main[n_hold], errors="coerce").clip(lower=0)))
        overlay_cols.append("n_holdings_scaled")

    for c in asset_cols:
        main[c] = pd.to_numeric(main[c], errors="coerce").fillna(0.0)
    if asset_cols and cfg["actions"].get("normalize_asset_weights", True):
        arr = main[asset_cols].clip(lower=0.0).to_numpy(dtype=float)
        denom = arr.sum(axis=1, keepdims=True)
        denom[denom <= 0] = 1.0
        main[asset_cols] = arr / denom

    for c in overlay_cols:
        main[c] = pd.to_numeric(main[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    action_cols = asset_cols + overlay_cols
    if not action_cols:
        raise ValueError("No action columns could be constructed. Check asset exposure and summary columns.")

    main["_non_missing_count"] = main.notna().sum(axis=1)
    main = main.sort_values([entity_col, date_col, "_non_missing_count"]).drop_duplicates([entity_col, date_col], keep="last")
    main = main.drop(columns=["_non_missing_count"])
    main = main.sort_values([entity_col, date_col]).reset_index(drop=True)

    if net_assets_col in main.columns:
        main["log_net_assets"] = np.log1p(pd.to_numeric(main[net_assets_col], errors="coerce").clip(lower=0)).fillna(0.0)
    if net_assets_col in main.columns:
        for flow_col in ["net_flow_3m", "sales_flow_3m", "redemption_flow_3m"]:
            if flow_col in main.columns:
                scaled_col = f"{flow_col}_scaled"
                main[scaled_col] = safe_divide(main[flow_col], main[net_assets_col], default=0.0)
                main[scaled_col] = pd.Series(main[scaled_col], index=main.index).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
                main[f"lag_{scaled_col}"] = main.groupby(entity_col)[scaled_col].shift(1)
    for c in action_cols:
        main[f"lag_{c}"] = main.groupby(entity_col)[c].shift(1)
    main["n_obs_entity"] = main.groupby(entity_col)[entity_col].transform("size")

    train_end = pd.Timestamp(cfg["sample"].get("train_end", "2023-12-31"))
    calib_end = pd.Timestamp(cfg["sample"].get("calib_end", "2024-12-31"))
    main["split"] = np.select(
        [main[date_col] <= train_end, main[date_col] <= calib_end],
        ["train", "calib"],
        default="test",
    )

    min_hist = int(cfg["sample"].get("min_history_obs", 1))
    main = main.loc[main["n_obs_entity"] >= min_hist].copy()
    main = main.dropna(subset=[entity_col, date_col])

    seed = int(cfg["sample"].get("random_seed", 42))
    rng = np.random.default_rng(seed)
    parts = []
    for split, max_key in [("train", "max_train_obs"), ("test", "max_test_obs")]:
        tmp = main[main["split"] == split]
        max_n = cfg["sample"].get(max_key)
        if max_n is not None and len(tmp) > int(max_n):
            idx = rng.choice(tmp.index.to_numpy(), size=int(max_n), replace=False)
            parts.append(main.loc[idx])
        else:
            parts.append(tmp)
    parts.append(main[main["split"] == "calib"])
    main = pd.concat(parts, axis=0).drop_duplicates([entity_col, date_col]).sort_values([entity_col, date_col]).reset_index(drop=True)

    main = assign_clusters(main, cfg, action_cols)
    state_cols = infer_state_cols(main, cfg, action_cols, entity_col, date_col)

    if out_dir:
        out = Path(out_dir)
        smart_write_table(main, out / "panel_model_ready.parquet")
        if action_cleaning_reports and cfg.get("actions", {}).get("summary_action_cleaning", {}).get("write_report", True):
            smart_write_table(pd.DataFrame(action_cleaning_reports), out / "action_cleaning_report.csv")
        # Record keyword prevalence separately from merge coverage.
        adv_label_cols = list(dict.fromkeys(cfg.get("adv_contract_constraints", {}).get("label_cols", []) or []))
        adv_label_cols = [c for c in adv_label_cols if c in main.columns]
        adv_rows = []
        if "adv_contract_merge_rate" in main.columns:
            adv_rows.append({
                "metric": "adv_contract_merge_rate",
                "value": float(pd.to_numeric(main["adv_contract_merge_rate"], errors="coerce").dropna().iloc[0]) if pd.to_numeric(main["adv_contract_merge_rate"], errors="coerce").notna().any() else np.nan,
                "n_rows": len(main),
            })
        if "adv_contract_merge_method" in main.columns:
            methods = main["adv_contract_merge_method"].astype(str).value_counts(dropna=False)
            for method, count in methods.items():
                adv_rows.append({"metric": f"adv_contract_merge_method:{method}", "value": int(count), "n_rows": len(main)})
        for c in adv_label_cols:
            x = _binary_or_numeric(main[c])
            adv_rows.append({
                "metric": f"adv_label_nonzero_rate_{c}",
                "value": float((x.fillna(0.0) > 0).mean()),
                "n_nonzero": int((x.fillna(0.0) > 0).sum()),
                "n_rows": len(main),
            })
        if adv_rows:
            smart_write_table(pd.DataFrame(adv_rows), out / "adv_contract_label_signal_report.csv")
        write_json({"action_cols": action_cols, "asset_action_cols": asset_cols, "overlay_action_cols": overlay_cols, "state_cols": state_cols}, out / "action_columns.json")

    return PanelData(panel=main, action_cols=action_cols, asset_action_cols=asset_cols, overlay_action_cols=overlay_cols, state_cols=state_cols, entity_col=entity_col, date_col=date_col)


def assign_clusters(main: pd.DataFrame, cfg: dict[str, Any], action_cols: list[str]) -> pd.DataFrame:
    """Assign mandate clusters without sklearn.

    Clusters use exogenous governance and ADV features, fitted on the training
    split with NumPy Lloyd k-means. If no features are available, all rows enter
    one cluster.
    """
    df = main.copy()
    feature_candidates = cfg.get("clusters", {}).get("feature_candidates", [])
    features = present_cols(df, feature_candidates)
    if not features:
        df["mandate_cluster"] = 0
        return df
    train = df[df["split"] == "train"].copy()
    if len(train) < 10:
        df["mandate_cluster"] = 0
        return df
    n_clusters = int(cfg.get("clusters", {}).get("n_clusters", 16))
    min_cluster_size = max(1, int(cfg.get("clusters", {}).get("min_cluster_size", 200)))
    n_clusters = max(1, min(n_clusters, max(1, len(train) // min_cluster_size)))
    if n_clusters <= 1:
        df["mandate_cluster"] = 0
        return df

    X_train_raw = _numeric_feature_frame(train, features)
    X_all_raw = _numeric_feature_frame(df, features)
    med = X_train_raw.median(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_train = X_train_raw.replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(dtype=float)
    X_all = X_all_raw.replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(dtype=float)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-12] = 1.0
    X_train = (X_train - mean) / std
    X_all = (X_all - mean) / std
    rng = np.random.default_rng(int(cfg["sample"].get("random_seed", 42)))
    centers = _fit_kmeans_numpy(X_train, n_clusters=n_clusters, rng=rng, max_iter=30)
    df["mandate_cluster"] = _predict_kmeans_numpy(X_all, centers, chunk=50000).astype(int)
    return df


def _fit_kmeans_numpy(X: np.ndarray, n_clusters: int, rng: np.random.Generator, max_iter: int = 30) -> np.ndarray:
    n, d = X.shape
    if n_clusters <= 1 or n == 0:
        return np.zeros((1, d), dtype=float)
    init_idx = rng.choice(np.arange(n), size=min(n_clusters, n), replace=False)
    centers = X[init_idx].copy()
    if centers.shape[0] < n_clusters:
        pad = np.repeat(centers[:1], n_clusters - centers.shape[0], axis=0)
        centers = np.vstack([centers, pad])
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        new_labels = _predict_kmeans_numpy(X, centers, chunk=50000)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centers[k] = X[mask].mean(axis=0)
            else:
                centers[k] = X[int(rng.integers(0, n))]
    return centers


def _predict_kmeans_numpy(X: np.ndarray, centers: np.ndarray, chunk: int = 50000) -> np.ndarray:
    labels = np.empty(X.shape[0], dtype=int)
    c2 = np.sum(centers * centers, axis=1)
    for start in range(0, X.shape[0], chunk):
        end = min(start + chunk, X.shape[0])
        xx = X[start:end]
        dist = np.sum(xx * xx, axis=1, keepdims=True) - 2 * xx @ centers.T + c2[None, :]
        labels[start:end] = np.argmin(dist, axis=1)
    return labels



def _simple_kmeans_fit(X: np.ndarray, n_clusters: int, rng: np.random.Generator, max_iter: int = 40) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) == 0:
        return np.zeros((1, 1), dtype=float)
    n_clusters = int(max(1, min(n_clusters, len(X))))
    init_idx = rng.choice(len(X), size=n_clusters, replace=False)
    centers = X[init_idx].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(max_iter):
        new_labels = _simple_kmeans_predict(X, centers, batch_size=20000)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centers[k] = X[mask].mean(axis=0)
            else:
                centers[k] = X[int(rng.integers(0, len(X)))]
    return centers


def _simple_kmeans_predict(X: np.ndarray, centers: np.ndarray, batch_size: int = 50000) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    labels = np.zeros(len(X), dtype=int)
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        diff = X[start:end, None, :] - centers[None, :, :]
        dist = np.einsum("bkd,bkd->bk", diff, diff)
        labels[start:end] = np.argmin(dist, axis=1)
    return labels

def _numeric_feature_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in features:
        s = df[c]
        if s.dtype == bool:
            out[c] = s.astype(float)
        elif pd.api.types.is_numeric_dtype(s):
            out[c] = pd.to_numeric(s, errors="coerce")
        else:
            lower = s.astype(str).str.lower()
            truthy = lower.isin(["true", "yes", "y", "1", "t"])
            falsy = lower.isin(["false", "no", "n", "0", "f", "nan", "none", ""])
            if (truthy | falsy).mean() > 0.8:
                out[c] = np.where(truthy, 1.0, np.where(falsy, 0.0, np.nan))
            else:
                codes, _ = pd.factorize(s.astype(str), sort=True)
                out[c] = codes.astype(float)
    return out


def infer_state_cols(df: pd.DataFrame, cfg: dict[str, Any], action_cols: list[str], entity_col: str, date_col: str) -> list[str]:
    excluded = set(action_cols + [f"lag_{c}" for c in action_cols] + [entity_col, date_col, "split"])
    state_cols = []
    for c in df.columns:
        lc = c.lower()
        if c in excluded:
            continue
        if lc in {"accession_number", "registrant_cik", "filing_id", "advisor_id", "crd_number"}:
            continue
        if "ret_next" in lc or lc.endswith("_next"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == bool:
            state_cols.append(c)
    return state_cols


def audit_panel(pdta: PanelData, out_dir: str | Path) -> pd.DataFrame:
    df = pdta.panel
    rows = []
    rows.append({"metric": "n_rows", "value": len(df)})
    rows.append({"metric": "n_entities", "value": df[pdta.entity_col].nunique()})
    rows.append({"metric": "n_report_dates", "value": df[pdta.date_col].nunique()})
    for split in ["train", "calib", "test"]:
        rows.append({"metric": f"n_rows_{split}", "value": int((df["split"] == split).sum())})
    rows.append({"metric": "n_action_cols", "value": len(pdta.action_cols)})
    rows.append({"metric": "n_asset_action_cols", "value": len(pdta.asset_action_cols)})
    rows.append({"metric": "n_overlay_action_cols", "value": len(pdta.overlay_action_cols)})
    if "adv_contract_merge_rate" in df.columns:
        rows.append({"metric": "adv_contract_merge_rate", "value": float(pd.to_numeric(df["adv_contract_merge_rate"], errors="coerce").max())})
    if "adv_contract_merge_method" in df.columns and len(df):
        mode = df["adv_contract_merge_method"].astype(str).mode()
        rows.append({"metric": "adv_contract_merge_method", "value": mode.iloc[0] if len(mode) else ""})
    for c in pdta.action_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        rows.append({"metric": f"action_missing_{c}", "value": float(x.isna().mean())})
        rows.append({"metric": f"action_mean_{c}", "value": float(x.mean())})
        rows.append({"metric": f"action_min_{c}", "value": float(x.min(skipna=True)) if x.notna().any() else np.nan})
        rows.append({"metric": f"action_max_{c}", "value": float(x.max(skipna=True)) if x.notna().any() else np.nan})

    # Record nonzero ADV label rates alongside merge coverage.
    label_cols = [c for c in [
        "allows_or_discusses_derivatives", "shorting", "leverage",
        "liquidity_or_redemption_constraint", "discretionary_management", "wrap_fee",
        "performance_fee", "esg", "crypto", "legal_disciplinary", "custody",
    ] if c in df.columns]
    label_rows = []
    for c in label_cols:
        x = _binary_or_numeric(df[c])
        label_rows.append({
            "label": c,
            "n": int(len(x)),
            "missing_rate": float(x.isna().mean()),
            "nonzero_count": int((x.fillna(0.0) > 0).sum()),
            "nonzero_rate": float((x.fillna(0.0) > 0).mean()),
            "mean": float(x.fillna(0.0).mean()),
        })
        rows.append({"metric": f"adv_label_nonzero_rate_{c}", "value": float((x.fillna(0.0) > 0).mean())})
    out = pd.DataFrame(rows)
    smart_write_table(out, Path(out_dir) / "sample_audit_report.csv")
    if label_rows:
        smart_write_table(pd.DataFrame(label_rows), Path(out_dir) / "adv_contract_label_audit.csv")
    return out
