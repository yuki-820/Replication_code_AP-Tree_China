"""
Factor-model tests and paper-ready result tables for pruned Section-SDF returns.

Main outputs
------------
1. All_Regression_Results.csv
   Compact long-format audit table for every Section-SDF x factor model.

2. All_Section_SDF_Results_Wide.csv
   Main wide-format result table with performance metrics, best pruning
   hyperparameters, factor-model alpha, time-series R2, and cleaned-sample XS-R².

3. Top10_AllModelsSignificant_Full.csv
4. Top10_AllModelsSignificant_Cleaned.csv
   Full and cleaned samples are ranked independently.

5. XS_R²_Cleaned_Sample.csv
   XS-R² for all cleaned-sample strategies, including TripleSort64_clean.

6. LongOnly_Sharpe_Summary.csv
   Long-only strategies: performance summary only; no factor regressions.

7. Top_SDF_Weight_Distributions.csv
   Selected-node weights for:
     - top 3 Full-sample SDFs by monthly out-of-sample Sharpe;
     - top 6 Cleaned-sample SDFs by monthly out-of-sample Sharpe.

8. Factor_Test_Run_Manifest.csv
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CONFIG = {
    "factor_dir": PROJECT_ROOT / "data" / "factors",
    "output_dir": PROJECT_ROOT / "output" / "tables",
    "nw_lags": None,
    "significance_level": 0.05,
    "require_positive_alpha": False,
    "top_n": 10,
    "min_observations": 12,
    "top_full_sdfs": 3,
    "top_cleaned_sdfs": 6,
    "models": {
        "CH3": {
            "file": "CH3_factors_monthly_202602.xlsx",
            "sheet": "Returnseries",
            "factors": {
                "Mkt-RF": ["mktrf", "MKT_RF", "Mkt-RF"],
                "SMB": ["SMB", "smb"],
                "VMG": ["VMG", "vmg"],
            },
        },
        "CH4": {
            "file": "CH4_factors_monthly_202602.xlsx",
            "sheet": "Returnseries",
            "factors": {
                "Mkt-RF": ["mktrf", "MKT_RF", "Mkt-RF"],
                "SMB": ["SMB", "smb"],
                "VMG": ["VMG", "vmg"],
                "PMO": ["PMO", "pmo"],
            },
        },
        "FF5": {
            "file": "fivefactor_monthly.csv",
            "sheet": None,
            "factors": {
                "Mkt-RF": ["mkt_rf", "MKT_RF", "Mkt-RF"],
                "SMB": ["smb", "SMB"],
                "HML": ["hml", "HML"],
                "RMW": ["rmw", "RMW"],
                "CMA": ["cma", "CMA"],
            },
        },
        "FF6": {
            "file": "fivefactor_monthly.csv",
            "sheet": None,
            "factors": {
                "Mkt-RF": ["mkt_rf", "MKT_RF", "Mkt-RF"],
                "SMB": ["smb", "SMB"],
                "HML": ["hml", "HML"],
                "RMW": ["rmw", "RMW"],
                "CMA": ["cma", "CMA"],
                "UMD": ["umd", "UMD", "mom", "MOM"],
            },
        },
        "Carhart4": {
            "file": "fivefactor_monthly.csv",
            "sheet": None,
            "factors": {
                "Mkt-RF": ["mkt_rf", "MKT_RF", "Mkt-RF"],
                "SMB": ["smb", "SMB"],
                "HML": ["hml", "HML"],
                "UMD": ["umd", "UMD", "mom", "MOM"],
            },
        },
        "Q5": {
            "file": "q5factor.csv",
            "sheet": 0,
            "factors": {
                "Mkt-RF": ["RmRf"],
                "ME": ["R_ME"],
                "IA": ["R_IA"],
                "ROE": ["R_ROE"],
                "EG": ["EG_factor"],
            },
        },
    },
    "strategies": {
        "TripleSort64": {
            "sample": "Full",
            "model_name": "TripleSort64",
            "path": (
                "output/pruned/TripleSort64_full/"
                "Test_Excess_Returns_TripleSort64.csv"
            ),
            "detail_path": (
                "output/pruned/TripleSort64_full/"
                "All_TripleSort_Detailed_Results.csv"
            ),
        },
        "APTree_K20": {
            "sample": "Full",
            "model_name": "AP-Tree",
            "path": (
                "output/pruned/AP-Tree_full/"
                "Test_Excess_Returns_kmax20.csv"
            ),
            "detail_path": (
                "output/pruned/AP-Tree_full/"
                "All_Sections_Detailed_Results_kmax20.csv"
            ),
        },
        "APTree_K40": {
            "sample": "Full",
            "model_name": "AP-Tree",
            "path": (
                "output/pruned/AP-Tree_full/"
                "Test_Excess_Returns_kmax40.csv"
            ),
            "detail_path": (
                "output/pruned/AP-Tree_full/"
                "All_Sections_Detailed_Results_kmax40.csv"
            ),
        },
        "TripleSort64_clean": {
            "sample": "Cleaned",
            "model_name": "TripleSort64",
            "path": (
                "output/pruned/TripleSort64_cleaned/"
                "Test_Excess_Returns_TripleSort64.csv"
            ),
            "detail_path": (
                "output/pruned/TripleSort64_cleaned/"
                "All_TripleSort_Detailed_Results.csv"
            ),
            "selected_node_path": (
                "output/pruned/TripleSort64_cleaned/"
                "Selected_Node_Test_Excess_Returns_TripleSort64.csv"
            ),
            "selected_node_header_levels": 2,
        },
        "APTree_clean_K20": {
            "sample": "Cleaned",
            "model_name": "AP-Tree",
            "path": (
                "output/pruned/AP-Tree_cleaned/"
                "Test_Excess_Returns_kmax20.csv"
            ),
            "detail_path": (
                "output/pruned/AP-Tree_cleaned/"
                "All_Sections_Detailed_Results_kmax20.csv"
            ),
            "selected_node_path": (
                "output/pruned/AP-Tree_cleaned/"
                "Selected_Node_Test_Excess_Returns_kmax20.csv"
            ),
            "selected_node_header_levels": 3,
        },
        "APTree_clean_K40": {
            "sample": "Cleaned",
            "model_name": "AP-Tree",
            "path": (
                "output/pruned/AP-Tree_cleaned/"
                "Test_Excess_Returns_kmax40.csv"
            ),
            "detail_path": (
                "output/pruned/AP-Tree_cleaned/"
                "All_Sections_Detailed_Results_kmax40.csv"
            ),
            "selected_node_path": (
                "output/pruned/AP-Tree_cleaned/"
                "Selected_Node_Test_Excess_Returns_kmax40.csv"
            ),
            "selected_node_header_levels": 3,
        },
        "TripleSort64_longonly": {
            "sample": "LongOnly",
            "model_name": "TripleSort64",
            "path": (
                "output/pruned/TripleSort64_cleaned_longonly/"
                "Test_Excess_Returns_TripleSort64_longonly.csv"
            ),
        },
        "APTree_longonly_K5": {
            "sample": "LongOnly",
            "model_name": "AP-Tree",
            "path": (
                "output/pruned/AP-Tree_cleaned_longonly/"
                "Test_Excess_Returns_kmax5.csv"
            ),
        },
    },
}

FEATURE_MAP = {
    "mkt_cap": "LME",
    "turnover": "Lturnover",
    "st_rev": "ST_Rev",
    "r12_2": "r12_2",
    "lt_rev": "LT_Rev",
    "idio_vol": "IdioVol",
    "fin_investment": "Fin_Investment",
    "fin_op": "Fin_OP",
    "fin_ac": "Fin_AC",
    "beme": "BEME",
}

PERFORMANCE_COLUMNS = [
    "Annualized_Sharpe",
    "Monthly_Sharpe",
    "Max_Drawdown",
    "Monthly_Avg_Excess_Return",
]

IDENTIFIER_COLUMNS = [
    "Sample",
    "Strategy",
    "Model_Name",
    "Weight_Scheme",
    "Section_SDF",
    "Section",
    "Feature",
]

# =============================================================================
# Generic helpers
# =============================================================================

def write_csv(df: pd.DataFrame, filename: str) -> None:
    """Write CSV using UTF-8 with BOM for spreadsheet compatibility."""
    output_path = Path(CONFIG["output_dir"]) / filename
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")

def find_column(columns: pd.Index, aliases: list[str]) -> str | None:
    """Find a column using case-insensitive exact aliases."""
    lookup = {str(column).strip().lower(): column for column in columns}

    for alias in aliases:
        matched = lookup.get(str(alias).strip().lower())
        if matched is not None:
            return matched

    return None

def parse_float(value: object) -> float:
    """Convert ordinary numeric or percentage-like text values to float."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if not value:
            return np.nan
        if value.endswith("%"):
            return pd.to_numeric(value[:-1], errors="coerce") / 100.0

    return pd.to_numeric(value, errors="coerce")

def parse_factor_dates(values: pd.Series, date_column: str) -> pd.Series:
    """Parse YYYYMMDD, YYYYMM, Excel-like, and ordinary date representations."""
    text = values.astype(str).str.replace(r"\.0$", "", regex=True)

    if str(date_column).strip().lower() == "trdmn":
        parsed = pd.to_datetime(text, format="%Y%m", errors="coerce")
    else:
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        parsed = parsed.where(
            parsed.notna(),
            pd.to_datetime(values, errors="coerce"),
        )
        parsed = parsed.where(
            parsed.notna(),
            pd.to_datetime(text, format="%Y%m", errors="coerce"),
        )

    return parsed

def load_factors(model_name: str) -> pd.DataFrame:
    """Load one factor model and enforce its complete factor schema."""
    cfg = CONFIG["models"][model_name]
    file_path = Path(CONFIG["factor_dir"]) / cfg["file"]

    if not file_path.exists():
        raise FileNotFoundError(
            f"{model_name}: factor file not found: {file_path}"
        )

    if file_path.suffix.lower() == ".csv":
        raw = pd.read_csv(file_path)
    else:
        raw = pd.read_excel(file_path, sheet_name=cfg["sheet"])

    date_column = find_column(
        raw.columns,
        ["日期_Date", "mnthdt", "trdmn", "date", "Date", "month", "Month"],
    )
    if date_column is None:
        raise ValueError(
            f"{model_name}: date column not found in {file_path.name}. "
            f"Available columns: {raw.columns.tolist()}"
        )

    selected_columns = {}
    missing_factors = []

    for canonical_name, aliases in cfg["factors"].items():
        source_name = find_column(raw.columns, aliases)

        if source_name is None:
            missing_factors.append(
                f"{canonical_name}: aliases={aliases}"
            )
        else:
            selected_columns[canonical_name] = source_name

    if missing_factors:
        raise ValueError(
            f"{model_name}: required factor columns are missing in "
            f"{file_path.name}.\n"
            f"Missing: {missing_factors}\n"
            f"Available columns: {raw.columns.tolist()}"
        )

    df = raw[[date_column] + list(selected_columns.values())].copy()
    df[date_column] = parse_factor_dates(df[date_column], str(date_column))
    df = df.dropna(subset=[date_column])
    df["year_month"] = df[date_column].dt.strftime("%Y-%m")
    df = df.set_index("year_month")

    df = df.rename(
        columns={
            source: canonical
            for canonical, source in selected_columns.items()
        }
    )

    df = df[list(cfg["factors"])].apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated(keep="first")].sort_index()

    if df.empty:
        raise ValueError(
            f"{model_name}: no valid monthly factor observations after loading."
        )

    return df

def normalize_monthly_index(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a date-like index to YYYY-MM, unique and sorted."""
    index_dates = pd.to_datetime(df.index, errors="coerce")
    valid = index_dates.notna()

    df = df.loc[valid].copy()
    df.index = index_dates[valid].strftime("%Y-%m")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated(keep="first")].sort_index()

    return df

def load_return_file(file_path: Path) -> pd.DataFrame:
    """Load monthly Section-SDF excess return data."""
    return normalize_monthly_index(
        pd.read_csv(file_path, index_col=0)
    )

def load_selected_node_return_file(
    file_path: Path,
    header_levels: int,
) -> pd.DataFrame:
    """Load selected-node returns with their MultiIndex column structure."""
    if header_levels not in {2, 3}:
        raise ValueError(
            f"Unsupported selected-node header level count: {header_levels}"
        )

    df = pd.read_csv(
        file_path,
        header=list(range(header_levels)),
        index_col=0,
    )

    df = normalize_monthly_index(df)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    return df

def extract_weight_scheme(portfolio_name: str) -> str:
    """Extract AP-Tree depth-weight scheme from a Section-SDF name."""
    match = re.search(
        r"_dw_power(\d+(?:\.\d+)?)$",
        str(portfolio_name),
    )

    return f"dw_power{match.group(1)}" if match else "none"

def base_section_name(portfolio_name: str) -> str:
    """Remove strategy-specific suffixes from a Section-SDF name."""
    name = re.sub(
        r"_dw_power\d+(?:\.\d+)?$",
        "",
        str(portfolio_name),
    )

    return re.sub(r"_triplesort$", "", name)

def extract_feature_string(portfolio_name: str) -> str:
    """Parse feature labels without breaking names containing underscores."""
    name = re.sub(
        r"^Sec\d+_?",
        "",
        base_section_name(portfolio_name),
    )

    remaining = name
    features = []

    for raw_feature in sorted(FEATURE_MAP, key=len, reverse=True):
        pattern = rf"(?:^|_){re.escape(raw_feature)}(?:_|$)"

        if re.search(pattern, remaining):
            features.append(FEATURE_MAP[raw_feature])
            remaining = re.sub(pattern, "_", remaining)

    return ", ".join(features) if features else base_section_name(portfolio_name)

def max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum peak-to-trough drawdown from compounded returns."""
    clean = returns.dropna().astype(float)

    if clean.empty:
        return np.nan

    wealth = (1.0 + clean).cumprod()
    drawdowns = wealth.div(wealth.cummax()).sub(1.0)

    return float(drawdowns.min())

def calculate_performance_metrics(returns: pd.Series) -> dict:
    """Calculate out-of-sample monthly performance statistics."""
    clean = returns.dropna().astype(float)
    n_obs = len(clean)

    if n_obs == 0:
        return {
            "Test_Nobs": 0,
            "Monthly_Avg_Excess_Return": np.nan,
            "Monthly_Sharpe": np.nan,
            "Annualized_Sharpe": np.nan,
            "Max_Drawdown": np.nan,
        }

    monthly_mean = clean.mean()
    monthly_volatility = clean.std(ddof=1) if n_obs > 1 else np.nan

    monthly_sharpe = (
        monthly_mean / monthly_volatility
        if pd.notna(monthly_volatility) and monthly_volatility > 0
        else np.nan
    )

    return {
        "Test_Nobs": n_obs,
        "Monthly_Avg_Excess_Return": monthly_mean,
        "Monthly_Sharpe": monthly_sharpe,
        "Annualized_Sharpe": (
            monthly_sharpe * np.sqrt(12)
            if pd.notna(monthly_sharpe)
            else np.nan
        ),
        "Max_Drawdown": max_drawdown(clean),
    }

# =============================================================================
# Pruning-result metadata and selected-node weights
# =============================================================================

def detail_section_candidates(
    strategy: str,
    portfolio: str,
) -> set[str]:
    """Return possible section labels used in pruning detailed-result files."""
    if strategy.startswith("TripleSort64"):
        return {
            str(portfolio),
            base_section_name(portfolio),
            f"{base_section_name(portfolio)}_triplesort",
        }

    return {
        str(portfolio),
        base_section_name(portfolio),
    }

def load_pruning_metadata(
    strategy: str,
    portfolio: str,
) -> dict:
    """
    Recover pruning metrics and best lambda values from detailed prune outputs.

    Section-SDF returns files do not contain lambda0/lambda2, selected-node
    count, or the actual pruning result values. These are recovered from the
    Type == 'Section Info' row of the detailed pruning output.
    """
    cfg = CONFIG["strategies"][strategy]
    detail_relative_path = cfg.get("detail_path")

    empty = {
        "Best_Lambda0": np.nan,
        "Best_Lambda2": np.nan,
        "Selected_Nodes": np.nan,
        "Prune_Monthly_Sharpe": np.nan,
        "Prune_Annualized_Sharpe": np.nan,
        "Prune_Max_Drawdown": np.nan,
        "Prune_Monthly_Avg_Excess_Return": np.nan,
    }

    if not detail_relative_path:
        return empty

    detail_path = PROJECT_ROOT / detail_relative_path

    if not detail_path.exists():
        print(
            f"Warning: detailed pruning file not found for {strategy}: "
            f"{detail_path}"
        )
        return empty

    detailed = pd.read_csv(detail_path)

    if "Section" not in detailed.columns:
        print(
            f"Warning: {detail_path.name} has no 'Section' column."
        )
        return empty

    section_rows = detailed.copy()

    if "Type" in section_rows.columns:
        section_rows = section_rows.loc[
            section_rows["Type"].astype(str).eq("Section Info")
        ]

    candidates = detail_section_candidates(strategy, portfolio)

    section_rows = section_rows.loc[
        section_rows["Section"].astype(str).isin(candidates)
    ]

    weight_scheme = extract_weight_scheme(portfolio)

    if "Weight_Scheme" in section_rows.columns:
        section_rows = section_rows.loc[
            section_rows["Weight_Scheme"].astype(str).eq(weight_scheme)
        ]

    if section_rows.empty:
        return empty

    row = section_rows.iloc[0]

    def read_value(*column_names: str) -> float:
        for column_name in column_names:
            if column_name in row.index:
                return parse_float(row[column_name])
        return np.nan

    return {
        "Best_Lambda0": read_value(
            "Best_lambda0",
            "Best_λ0",
            "Best_Lambda0",
        ),
        "Best_Lambda2": read_value(
            "Best_lambda2",
            "Best_λ2",
            "Best_Lambda2",
        ),
        "Selected_Nodes": read_value(
            "Selected_Portfolios",
            "Selected_Nodes",
        ),
        "Prune_Monthly_Sharpe": read_value(
            "Test_Monthly_Sharpe",
        ),
        "Prune_Annualized_Sharpe": read_value(
            "Annualised_Sharpe",
            "Annualized_Sharpe",
        ),
        "Prune_Max_Drawdown": read_value(
            "Max_Drawdown",
        ),
        "Prune_Monthly_Avg_Excess_Return": read_value(
            "Monthly_Avg_Excess_Return",
        ),
    }

def load_selected_weights(
    strategy: str,
    portfolio: str,
) -> pd.DataFrame:
    """Load selected node weights for one Section-SDF."""
    cfg = CONFIG["strategies"][strategy]
    detail_relative_path = cfg.get("detail_path")

    if not detail_relative_path:
        return pd.DataFrame(columns=["Node_Name", "Weight", "Depth"])

    detail_path = PROJECT_ROOT / detail_relative_path

    if not detail_path.exists():
        print(
            f"Warning: detailed pruning file not found for {strategy}: "
            f"{detail_path}"
        )
        return pd.DataFrame(columns=["Node_Name", "Weight", "Depth"])

    detailed = pd.read_csv(detail_path)

    if "Type" not in detailed.columns or "Section" not in detailed.columns:
        print(
            f"Warning: incompatible detailed pruning file: {detail_path}"
        )
        return pd.DataFrame(columns=["Node_Name", "Weight", "Depth"])

    node_column = (
        "Node_Name"
        if "Node_Name" in detailed.columns
        else "Portfolio_Name"
    )

    if node_column not in detailed.columns or "Weight" not in detailed.columns:
        print(
            f"Warning: node or weight field missing from {detail_path.name}"
        )
        return pd.DataFrame(columns=["Node_Name", "Weight", "Depth"])

    candidates = detail_section_candidates(strategy, portfolio)

    selected = detailed.loc[
        detailed["Type"].astype(str).eq("Selected Node")
        & detailed["Section"].astype(str).isin(candidates)
    ].copy()

    weight_scheme = extract_weight_scheme(portfolio)

    if "Weight_Scheme" in selected.columns:
        selected = selected.loc[
            selected["Weight_Scheme"].astype(str).eq(weight_scheme)
        ]

    if selected.empty:
        return pd.DataFrame(columns=["Node_Name", "Weight", "Depth"])

    selected = selected.rename(
        columns={
            node_column: "Node_Name",
        }
    )

    selected["Weight"] = selected["Weight"].map(parse_float)

    if "Depth" not in selected.columns:
        selected["Depth"] = np.nan

    return selected[["Node_Name", "Weight", "Depth"]].dropna(
        subset=["Weight"]
    )

# =============================================================================
# Factor regressions
# =============================================================================

def run_factor_regression(
    strategy_returns: pd.Series,
    factors: pd.DataFrame,
) -> dict | None:
    """Estimate alpha with HAC inference and conventional OLS R²."""
    data = pd.concat(
        [strategy_returns.rename("return"), factors],
        axis=1,
        join="inner",
    ).dropna()

    if len(data) < CONFIG["min_observations"]:
        return None

    y = data["return"]
    x = sm.add_constant(data[factors.columns], has_constant="add")

    nw_lags = CONFIG["nw_lags"]

    if nw_lags is None:
        nw_lags = max(
            0,
            int(4 * (len(y) / 100.0) ** (2.0 / 9.0)),
        )

    try:
        ols = sm.OLS(y, x).fit()
        hac = sm.OLS(y, x).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": nw_lags},
        )
    except (ValueError, np.linalg.LinAlgError):
        return None

    return {
        "Alpha_monthly": hac.params["const"],
        "Alpha_annualized": hac.params["const"] * 12,
        "Alpha_t": hac.tvalues["const"],
        "Alpha_p": hac.pvalues["const"],
        "R2": ols.rsquared,
        "Regression_Nobs": len(y),
        "Newey_West_Lags": nw_lags,
    }

def is_significant(regression: pd.Series) -> bool:
    """Apply the configured alpha significance rule."""
    passed = (
        pd.notna(regression["Alpha_p"])
        and regression["Alpha_p"] < CONFIG["significance_level"]
    )

    if CONFIG["require_positive_alpha"]:
        passed = passed and regression["Alpha_monthly"] > 0

    return bool(passed)

# =============================================================================
# XS-R² for selected cleaned-sample basis nodes
# =============================================================================

def selected_node_column_metadata(
    strategy: str,
    column: tuple,
) -> tuple[str, str, str]:
    """
    Map a selected-node return column to section, depth strategy, and node name.

    TripleSort64 uses a two-level column index:
        (Section, Portfolio_Name)

    AP-Tree uses a three-level column index:
        (Section, Weight_Scheme, Node_Name)
    """
    if strategy == "TripleSort64_clean":
        section, node_name = column
        return str(section), "none", str(node_name)

    section, weight_scheme, node_name = column
    return str(section), str(weight_scheme), str(node_name)

def calculate_xs_r2(
    node_returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> dict:
    """
    Estimate node-level OLS alphas and calculate JF-style XS-R².

    XS-R² = 1 - [N / (N - K)] *
                 [sum(alpha_i²) / sum(mean_return_i²)]

    N is the number of selected basis nodes and K is the number of factors.
    """
    data = pd.concat(
        [node_returns, factors],
        axis=1,
        join="inner",
    ).dropna()

    n_assets = node_returns.shape[1]
    n_factors = factors.shape[1]

    result = {
        "Status": "ok",
        "N_Selected_Nodes": n_assets,
        "N_Model_Factors": n_factors,
        "XS_R²": np.nan,
    }

    if len(data) < CONFIG["min_observations"]:
        result["Status"] = "insufficient_observations"
        return result

    if n_assets <= n_factors:
        result["Status"] = "insufficient_nodes_for_degree_adjustment"
        return result

    returns = data.iloc[:, :n_assets]
    factor_data = data.iloc[:, n_assets:]
    x = sm.add_constant(factor_data, has_constant="add")

    alpha_values = []
    mean_return_values = []

    for node_name in returns.columns:
        try:
            fitted = sm.OLS(returns[node_name], x).fit()
        except (ValueError, np.linalg.LinAlgError):
            result["Status"] = "node_regression_failed"
            return result

        alpha_values.append(fitted.params["const"])
        mean_return_values.append(returns[node_name].mean())

    sum_alpha_squared = float(np.square(alpha_values).sum())
    sum_mean_return_squared = float(np.square(mean_return_values).sum())

    if sum_mean_return_squared <= 1e-16:
        result["Status"] = "zero_mean_return_denominator"
        return result

    result["XS_R²"] = float(
        1.0
        - (n_assets / (n_assets - n_factors))
        * (sum_alpha_squared / sum_mean_return_squared)
    )

    return result

def calculate_cleaned_xs_r2(
    factor_models: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Calculate XS-R² for every cleaned-sample strategy, Section, depth scheme,
    and configured factor model.
    """
    rows = []

    for strategy, strategy_cfg in CONFIG["strategies"].items():
        if strategy_cfg["sample"] != "Cleaned":
            continue

        selected_node_relative_path = strategy_cfg.get(
            "selected_node_path"
        )

        if not selected_node_relative_path:
            continue

        node_path = PROJECT_ROOT / selected_node_relative_path

        if not node_path.exists():
            print(
                f"Warning: selected-node file not found for {strategy}: "
                f"{node_path}"
            )
            continue

        selected_nodes = load_selected_node_return_file(
            node_path,
            strategy_cfg["selected_node_header_levels"],
        )

        grouped_columns: dict[tuple[str, str], list[tuple]] = {}

        for column in selected_nodes.columns:
            section, weight_scheme, _ = selected_node_column_metadata(
                strategy,
                column,
            )

            grouped_columns.setdefault(
                (section, weight_scheme),
                [],
            ).append(column)

        print(
            f"Calculating XS-R² for {strategy}: "
            f"{len(grouped_columns)} Section-SDF specifications"
        )

        for (section, weight_scheme), columns in grouped_columns.items():
            section_nodes = selected_nodes.loc[:, columns].copy()

            section_nodes.columns = [
                selected_node_column_metadata(strategy, column)[2]
                for column in columns
            ]

            if section_nodes.columns.duplicated().any():
                section_nodes = (
                    section_nodes.T.groupby(level=0).mean().T
                )

            for model_name, factor_df in factor_models.items():
                xs_result = calculate_xs_r2(
                    section_nodes,
                    factor_df,
                )

                rows.append({
                    "Sample": "Cleaned",
                    "Strategy": strategy,
                    "Model_Name": strategy_cfg["model_name"],
                    "Weight_Scheme": weight_scheme,
                    "Section": base_section_name(section),
                    "Section_SDF": section,
                    "Model": model_name,
                    **xs_result,
                })

    return pd.DataFrame(rows)

# =============================================================================
# Table construction
# =============================================================================

def build_wide_result(
    strategy: str,
    sample: str,
    portfolio: str,
    performance: dict,
    pruning_metadata: dict,
    model_results: dict[str, pd.DataFrame],
    model_names: list[str],
    xs_lookup: dict[tuple[str, str, str, str], float],
) -> dict:
    """Build one compact Section-SDF result row across all factor models."""
    weight_scheme = extract_weight_scheme(portfolio)
    section = base_section_name(portfolio)

    result = {
        "Sample": sample,
        "Strategy": strategy,
        "Model_Name": CONFIG["strategies"][strategy]["model_name"],
        "Weight_Scheme": weight_scheme,
        "Section_SDF": portfolio,
        "Section": section,
        "Feature": extract_feature_string(portfolio),
        "Annualized_Sharpe": pruning_metadata[
            "Prune_Annualized_Sharpe"
        ],
        "Monthly_Sharpe": pruning_metadata[
            "Prune_Monthly_Sharpe"
        ],
        "Max_Drawdown": pruning_metadata["Prune_Max_Drawdown"],
        "Monthly_Avg_Excess_Return": pruning_metadata[
            "Prune_Monthly_Avg_Excess_Return"
        ],
        "Best_Lambda0": pruning_metadata["Best_Lambda0"],
        "Best_Lambda2": pruning_metadata["Best_Lambda2"],
        "Selected_Nodes": pruning_metadata["Selected_Nodes"],
    }

    for metric_name in PERFORMANCE_COLUMNS:
        if pd.isna(result[metric_name]):
            result[metric_name] = performance[metric_name]

    all_models_significant = True

    for model_name in model_names:
        regression = model_results[model_name].loc[portfolio]

        result[f"{model_name}_Alpha_Annualized"] = (
            regression["Alpha_annualized"]
        )
        result[f"{model_name}_R²"] = regression["R2"]

        if sample == "Cleaned":
            xs_key = (
                strategy,
                section,
                weight_scheme,
                model_name,
            )
            result[f"{model_name}_XS_R²"] = xs_lookup.get(xs_key, np.nan)
        else:
            result[f"{model_name}_XS_R²"] = np.nan

        all_models_significant = (
            all_models_significant
            and is_significant(regression)
        )

    result["Passes_All_Models"] = all_models_significant

    return result

def paper_table_columns(model_names: list[str]) -> list[str]:
    """Return compact paper-ready column order for wide Section-SDF tables."""
    columns = [
        "Rank",
        "Sample",
        "Strategy",
        "Model_Name",
        "Weight_Scheme",
        "Section",
        "Feature",
        "Annualized_Sharpe",
        "Monthly_Sharpe",
        "Max_Drawdown",
        "Monthly_Avg_Excess_Return",
        "Best_Lambda0",
        "Best_Lambda2",
        "Selected_Nodes",
    ]

    for model_name in model_names:
        columns.extend([
            f"{model_name}_Alpha_Annualized",
            f"{model_name}_R²",
            f"{model_name}_XS_R²",
        ])

    return columns

def compact_table(
    df: pd.DataFrame,
    model_names: list[str],
    include_rank: bool = False,
) -> pd.DataFrame:
    """Select paper-ready columns that are present in a DataFrame."""
    columns = paper_table_columns(model_names)

    if not include_rank:
        columns = [
            column for column in columns
            if column != "Rank"
        ]

    return df.loc[
        :,
        [column for column in columns if column in df.columns],
    ].copy()

def build_long_only_sharpe_summary(
    returns_store: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build the single requested long-only performance summary."""
    rows = []

    for strategy, returns in returns_store.items():
        strategy_cfg = CONFIG["strategies"][strategy]

        if strategy_cfg["sample"] != "LongOnly":
            continue

        for portfolio in returns.columns:
            metrics = calculate_performance_metrics(
                returns[portfolio]
            )

            rows.append({
                "Strategy": strategy,
                "Model_Name": strategy_cfg["model_name"],
                "Weight_Scheme": extract_weight_scheme(portfolio),
                "Section": base_section_name(portfolio),
                "Feature": extract_feature_string(portfolio),
                "Annualized_Sharpe": metrics["Annualized_Sharpe"],
                "Monthly_Sharpe": metrics["Monthly_Sharpe"],
                "Max_Drawdown": metrics["Max_Drawdown"],
                "Monthly_Avg_Excess_Return": metrics[
                    "Monthly_Avg_Excess_Return"
                ],
            })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "Strategy",
                "Weight_Scheme",
                "Monthly_Sharpe",
            ],
            ascending=[True, True, False],
        )
        .reset_index(drop=True)
    )

def build_top_sdf_weight_distributions(
    wide_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Export selected-node weights for the globally highest-Sharpe SDFs.

    Full sample: top 3 SDFs across all Full-sample strategies.
    Cleaned sample: top 6 SDFs across all Cleaned-sample strategies.
    """
    output_rows = []

    ranking_rules = {
        "Full": CONFIG["top_full_sdfs"],
        "Cleaned": CONFIG["top_cleaned_sdfs"],
    }

    for sample, top_n in ranking_rules.items():
        candidates = wide_results.loc[
            wide_results["Sample"].eq(sample)
        ].copy()

        top_sdfs = (
            candidates
            .sort_values("Monthly_Sharpe", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        for rank, sdf in enumerate(
            top_sdfs.itertuples(index=False),
            start=1,
        ):
            selected_weights = load_selected_weights(
                sdf.Strategy,
                sdf.Section_SDF,
            )

            if selected_weights.empty:
                print(
                    f"Warning: no selected weights found for "
                    f"{sdf.Strategy} / {sdf.Section_SDF}"
                )
                continue

            for node in selected_weights.itertuples(index=False):
                output_rows.append({
                    "Sample": sample,
                    "Rank": rank,
                    "Strategy": sdf.Strategy,
                    "Model_Name": sdf.Model_Name,
                    "Depth_Strategy": sdf.Weight_Scheme,
                    "Section_SDF": sdf.Section_SDF,
                    "Section": sdf.Section,
                    "Feature": sdf.Feature,
                    "Node_Name": node.Node_Name,
                    "Depth": node.Depth,
                    "Weight": node.Weight,
                    "Selected_Nodes": sdf.Selected_Nodes,
                    "Annualized_Sharpe": sdf.Annualized_Sharpe,
                    "Monthly_Sharpe": sdf.Monthly_Sharpe,
                    "Max_Drawdown": sdf.Max_Drawdown,
                    "Monthly_Avg_Excess_Return": (
                        sdf.Monthly_Avg_Excess_Return
                    ),
                    "Best_Lambda0": sdf.Best_Lambda0,
                    "Best_Lambda2": sdf.Best_Lambda2,
                })

    if not output_rows:
        return pd.DataFrame()

    return pd.DataFrame(output_rows).sort_values(
        ["Sample", "Rank", "Weight"],
        ascending=[True, True, False],
    )

# =============================================================================
# Main procedure
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("Step 4: Factor Model Tests and Paper-Ready Tables")
    print("=" * 80)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(CONFIG["models"])

    print("Loading factor models...")
    factor_models = {
        model_name: load_factors(model_name)
        for model_name in model_names
    }

    for model_name, factor_df in factor_models.items():
        print(
            f"  {model_name}: {len(factor_df)} months, "
            f"factors = {', '.join(factor_df.columns)}"
        )

    # -------------------------------------------------------------------------
    # 1. Load all available strategy return files, including long-only files.
    # -------------------------------------------------------------------------
    returns_store: dict[str, pd.DataFrame] = {}

    for strategy, strategy_cfg in CONFIG["strategies"].items():
        return_path = PROJECT_ROOT / strategy_cfg["path"]

        if not return_path.exists():
            print(
                f"Warning: return file not found for {strategy}: "
                f"{return_path}"
            )
            continue

        returns_store[strategy] = load_return_file(return_path)

        print(
            f"Loaded {strategy}: "
            f"{returns_store[strategy].shape[1]} Section-SDF returns"
        )

    # -------------------------------------------------------------------------
    # 2. XS-R² for all cleaned-sample strategies and all configured models.
    # -------------------------------------------------------------------------
    print("\nCalculating cleaned-sample XS-R²...")
    xs_r2_results = calculate_cleaned_xs_r2(factor_models)

    xs_r2_output_columns = [
        "Sample",
        "Strategy",
        "Model_Name",
        "Weight_Scheme",
        "Section",
        "Section_SDF",
        "Model",
        "N_Selected_Nodes",
        "N_Model_Factors",
        "XS_R²",
        "Status",
    ]

    write_csv(
        xs_r2_results.loc[
            :,
            [
                column for column in xs_r2_output_columns
                if column in xs_r2_results.columns
            ],
        ],
        "XS_R²_Cleaned_Sample.csv",
    )

    xs_lookup: dict[tuple[str, str, str, str], float] = {}

    if not xs_r2_results.empty:
        valid_xs = xs_r2_results.loc[
            xs_r2_results["Status"].eq("ok")
        ]

        for _, row in valid_xs.iterrows():
            xs_lookup[
                (
                    str(row["Strategy"]),
                    str(row["Section"]),
                    str(row["Weight_Scheme"]),
                    str(row["Model"]),
                )
            ] = row["XS_R²"]

    # -------------------------------------------------------------------------
    # 3. Time-series factor regressions for Full and Cleaned samples only.
    #    Long-only strategies intentionally receive no factor regressions.
    # -------------------------------------------------------------------------
    results_store: dict[str, dict[str, pd.DataFrame]] = {}
    all_regression_rows = []
    wide_rows = []

    for strategy, returns in returns_store.items():
        strategy_cfg = CONFIG["strategies"][strategy]
        sample = strategy_cfg["sample"]

        if sample == "LongOnly":
            continue

        print(
            f"\nRunning factor regressions: {strategy} "
            f"({returns.shape[1]} Section-SDFs)"
        )

        results_store[strategy] = {}

        performance_by_portfolio = {
            portfolio: calculate_performance_metrics(
                returns[portfolio]
            )
            for portfolio in returns.columns
        }

        pruning_metadata_by_portfolio = {
            portfolio: load_pruning_metadata(strategy, portfolio)
            for portfolio in returns.columns
        }

        for model_name, factor_df in factor_models.items():
            model_rows = []

            for portfolio in returns.columns:
                regression = run_factor_regression(
                    returns[portfolio],
                    factor_df,
                )

                if regression is None:
                    continue

                regression["Section_SDF"] = portfolio
                model_rows.append(regression)

                all_regression_rows.append({
                    "Sample": sample,
                    "Strategy": strategy,
                    "Model_Name": strategy_cfg["model_name"],
                    "Weight_Scheme": extract_weight_scheme(portfolio),
                    "Section_SDF": portfolio,
                    "Section": base_section_name(portfolio),
                    "Feature": extract_feature_string(portfolio),
                    "Model": model_name,
                    "Alpha_Monthly": regression[
                        "Alpha_monthly"
                    ],
                    "Alpha_Annualized": regression[
                        "Alpha_annualized"
                    ],
                    "Alpha_t": regression[
                        "Alpha_t"
                    ],
                    "Alpha_p": regression[
                        "Alpha_p"
                    ],
                    "Regression_Nobs": regression[
                        "Regression_Nobs"
                    ],
                    "Newey_West_Lags": regression[
                        "Newey_West_Lags"
                    ],
                    "R²": regression["R2"],
                    "XS_R²": xs_lookup.get(
                        (
                            strategy,
                            base_section_name(portfolio),
                            extract_weight_scheme(portfolio),
                            model_name,
                        ),
                        np.nan,
                    ),
                    "Annualized_Sharpe": performance_by_portfolio[
                        portfolio
                    ]["Annualized_Sharpe"],
                    "Monthly_Sharpe": performance_by_portfolio[
                        portfolio
                    ]["Monthly_Sharpe"],
                    "Max_Drawdown": performance_by_portfolio[
                        portfolio
                    ]["Max_Drawdown"],
                    "Monthly_Avg_Excess_Return": (
                        performance_by_portfolio[portfolio][
                            "Monthly_Avg_Excess_Return"
                        ]
                    ),
                    "Best_Lambda0": pruning_metadata_by_portfolio[
                        portfolio
                    ]["Best_Lambda0"],
                    "Best_Lambda2": pruning_metadata_by_portfolio[
                        portfolio
                    ]["Best_Lambda2"],
                    "Selected_Nodes": pruning_metadata_by_portfolio[
                        portfolio
                    ]["Selected_Nodes"],
                })

            if model_rows:
                results_store[strategy][model_name] = (
                    pd.DataFrame(model_rows)
                    .set_index("Section_SDF")
                )
                print(
                    f"  {model_name}: {len(model_rows)} regressions"
                )

        missing_models = [
            model_name
            for model_name in model_names
            if model_name not in results_store[strategy]
        ]

        if missing_models:
            print(
                f"Warning: {strategy} lacks results for: "
                f"{missing_models}"
            )
            continue

        common_portfolios = set.intersection(
            *(
                set(
                    results_store[strategy][model_name].index
                )
                for model_name in model_names
            )
        )

        for portfolio in sorted(common_portfolios):
            wide_rows.append(
                build_wide_result(
                    strategy=strategy,
                    sample=sample,
                    portfolio=portfolio,
                    performance=performance_by_portfolio[portfolio],
                    pruning_metadata=pruning_metadata_by_portfolio[
                        portfolio
                    ],
                    model_results=results_store[strategy],
                    model_names=model_names,
                    xs_lookup=xs_lookup,
                )
            )

    if not all_regression_rows:
        raise RuntimeError(
            "No Full or Cleaned sample factor regressions "
            "were successfully estimated."
        )

    if not wide_rows:
        raise RuntimeError(
            "No Section-SDF has complete results across all factor models."
        )

    regression_long = pd.DataFrame(all_regression_rows)
    wide_results = pd.DataFrame(wide_rows)

    # Detailed audit table: factor alpha, HAC inference, R², XS-R²,
    # performance, and lambdas.
    regression_long_columns = [
        "Sample",
        "Strategy",
        "Model_Name",
        "Weight_Scheme",
        "Section_SDF",
        "Section",
        "Feature",
        "Model",
        "Alpha_Monthly",
        "Alpha_Annualized",
        "Alpha_t",
        "Alpha_p",
        "Regression_Nobs",
        "Newey_West_Lags",
        "R²",
        "XS_R²",
        "Annualized_Sharpe",
        "Monthly_Sharpe",
        "Max_Drawdown",
        "Monthly_Avg_Excess_Return",
        "Best_Lambda0",
        "Best_Lambda2",
        "Selected_Nodes",
    ]

    write_csv(
        regression_long.loc[:, regression_long_columns],
        "All_Regression_Results.csv",
    )

    write_csv(
        compact_table(
            wide_results.sort_values(
                ["Sample", "Monthly_Sharpe"],
                ascending=[True, False],
            ),
            model_names,
        ),
        "All_Section_SDF_Results_Wide.csv",
    )

    # -------------------------------------------------------------------------
    # 4. Top 10: Full and Cleaned are ranked independently.
    # -------------------------------------------------------------------------
    all_model_significant = wide_results.loc[
        wide_results["Passes_All_Models"]
    ].copy()

    for sample in ("Full", "Cleaned"):
        ranked = (
            all_model_significant.loc[
                all_model_significant["Sample"].eq(sample)
            ]
            .sort_values("Monthly_Sharpe", ascending=False)
            .head(CONFIG["top_n"])
            .copy()
        )

        ranked.insert(
            0,
            "Rank",
            range(1, len(ranked) + 1),
        )

        write_csv(
            compact_table(
                ranked,
                model_names,
                include_rank=True,
            ),
            f"Top10_AllModelsSignificant_{sample}.csv",
        )

    # -------------------------------------------------------------------------
    # 5. Long-only: one compact Sharpe/performance summary only.
    # -------------------------------------------------------------------------
    long_only_summary = build_long_only_sharpe_summary(returns_store)

    write_csv(
        long_only_summary,
        "LongOnly_Sharpe_Summary.csv",
    )

    # -------------------------------------------------------------------------
    # 6. Top-SDF selected-node weight distributions for future 3D figures.
    # -------------------------------------------------------------------------
    top_sdf_weights = build_top_sdf_weight_distributions(
        wide_results
    )

    write_csv(
        top_sdf_weights,
        "Top_SDF_Weight_Distributions.csv",
    )

    # -------------------------------------------------------------------------
    # 7. Reproducibility manifest.
    # -------------------------------------------------------------------------
    manifest_rows = []

    for model_name, factors in factor_models.items():
        manifest_rows.append({
            "Model": model_name,
            "Factor_File": CONFIG["models"][model_name]["file"],
            "Factor_Columns_Used": ", ".join(factors.columns),
            "Factor_Start": factors.index.min(),
            "Factor_End": factors.index.max(),
            "Factor_Months": len(factors),
            "Alpha_Significance_Level": CONFIG[
                "significance_level"
            ],
            "Require_Positive_Alpha": CONFIG[
                "require_positive_alpha"
            ],
            "Newey_West_Lags": (
                "automatic"
                if CONFIG["nw_lags"] is None
                else CONFIG["nw_lags"]
            ),
            "Top_N_Per_Sample": CONFIG["top_n"],
        })

    write_csv(
        pd.DataFrame(manifest_rows),
        "Factor_Test_Run_Manifest.csv",
    )

    print("\n" + "=" * 80)
    print("Factor model tests completed.")
    print(f"Output directory: {output_dir}")
    print("Primary paper-ready outputs:")
    print("  All_Section_SDF_Results_Wide.csv")
    print("  Top10_AllModelsSignificant_Full.csv")
    print("  Top10_AllModelsSignificant_Cleaned.csv")
    print("  XS_R²_Cleaned_Sample.csv")
    print("  LongOnly_Sharpe_Summary.csv")
    print("  Top_SDF_Weight_Distributions.csv")
    print("=" * 80)

if __name__ == "__main__":
    main()
