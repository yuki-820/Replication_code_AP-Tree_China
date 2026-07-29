"""
Final publication-quality plotting script for the AP-Tree China study.

Required Step-4 outputs
-----------------------
output/tables/
    All_Regression_Results.csv
    XS_R²_Cleaned_Sample.csv
    Top_SDF_Weight_Distributions.csv

Optional turnover input
-----------------------
output/stock_weights_cleaned/
    Average_Turnover_Summary_Cleaned.csv

Main output
-----------
output/figures/main/
    02_Depth_Weight_Performance_Differences.pdf/.png
    03_Full_Sharpe_By_Section.pdf/.png
    03_Cleaned_Sharpe_By_Section.pdf/.png
    04_CH4_Factor_Evidence_By_Section.pdf/.png
    04_FF5_Factor_Evidence_By_Section.pdf/.png
    04_Q5_Factor_Evidence_By_Section.pdf/.png
    05_Turnover_By_Section.pdf/.png
    09_Full_TopSDF_WeightCube_Rank01.pdf/.png
    09_Full_TopSDF_WeightCube_Rank02.pdf/.png
    09_Full_TopSDF_WeightCube_Rank03.pdf/.png
    10_Cleaned_TopSDF_WeightCube_Rank01.pdf/.png
    ...
    10_Cleaned_TopSDF_WeightCube_Rank06.pdf/.png
    11_Main_Factor_Evidence_Combined.pdf/.png
    12_Full_Cleaned_Sharpe_Combined.pdf/.png
    13_Full_TopSDF_WeightCubes_Combined.pdf/.png
    14_Cleaned_TopSDF_WeightCubes_Combined.pdf/.png

Appendix output
---------------
output/figures/appendix/
    A01_Overall_Performance_Boxplots.pdf/.png
    A02_CH3_Factor_Evidence_By_Section.pdf/.png
    A03_FF6_Factor_Evidence_By_Section.pdf/.png
    A04_Carhart4_Factor_Evidence_By_Section.pdf/.png
    A05_Turnover_Distribution_Boxplot.pdf/.png
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.text import Text
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TABLE_DIR = PROJECT_ROOT / "output" / "tables"
TURNOVER_DIR = PROJECT_ROOT / "output" / "stock_weights_cleaned"

FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
MAIN_DIR = FIGURE_DIR / "main"
APPENDIX_DIR = FIGURE_DIR / "appendix"

PATHS = {
    "regression": TABLE_DIR / "All_Regression_Results.csv",
    "xs_r2": TABLE_DIR / "XS_R²_Cleaned_Sample.csv",
    "weight_distribution": TABLE_DIR / "Top_SDF_Weight_Distributions.csv",
    "turnover": (
        TURNOVER_DIR
        / "Average_Turnover_Summary_Cleaned.csv"
    ),
}

# =============================================================================
# Models, strategies, colours
# =============================================================================

MAIN_FACTOR_MODELS = [
    "CH4",
    "FF5",
    "Q5",
]

APPENDIX_FACTOR_MODELS = [
    "CH3",
    "FF6",
    "Carhart4",
]

MODEL_LABELS = {
    "CH3": "China Three-Factor Model",
    "CH4": "China Four-Factor Model",
    "FF5": "Fama-French Five-Factor Model",
    "FF6": "Fama-French Six-Factor Model",
    "Carhart4": "Carhart Four-Factor Model",
    "Q5": "q-Factor Model",
}

STRATEGY_ORDER = [
    "TripleSort64",
    "APTree_K20_dw_power0.5",
    "APTree_K20_dw_power2.0",
    "APTree_K40_dw_power0.5",
    "APTree_K40_dw_power2.0",
]

STRATEGY_LABELS = {
    "TripleSort64": "TripleSort64",
    "APTree_K20_dw_power0.5": "AP-Tree K=20\nDeep Boost",
    "APTree_K20_dw_power2.0": "AP-Tree K=20\nShallow Tilt",
    "APTree_K40_dw_power0.5": "AP-Tree K=40\nDeep Boost",
    "APTree_K40_dw_power2.0": "AP-Tree K=40\nShallow Tilt",
}

COLORS = {
    # Fixed strategy colours used consistently in every figure.
    "triple": "#00C853",        # TS64: bright green
    "ap20_deep": "#D50000",     # APK20 Deep Boost: vivid red
    "ap40_deep": "#FFD400",     # APK40 Deep Boost: bright yellow
    "ap20_shallow": "#FF4F87",  # APK20 Shallow Tilt: pink
    "ap40_shallow": "#FFF09A",  # APK40 Shallow Tilt: light yellow
    "positive": "#00C853",
    "negative": "#F20D0D",
    "line": "#A7A7A7",
    "grid": "#D9D9D9",
    "cube_grid": "#E3E3E3",
    "neutral": "#626262",
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

RAW_FEATURES = list(FEATURE_MAP.keys())

plt.rcParams.update({
    # Screen rendering remains responsive; exported files use 600 DPI.
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "lines.linewidth": 2.2,
    "lines.markersize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =============================================================================
# General utilities
# =============================================================================

def read_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    """Read a CSV safely, including UTF-8 BOM files."""
    if not path.exists():
        print(f"SKIP missing input: {path}")
        return None

    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, **kwargs)

def save(fig: plt.Figure, folder: Path, stem: str) -> None:
    """Save one high-resolution figure as both PDF and PNG."""
    folder.mkdir(parents=True, exist_ok=True)

    pdf_path = folder / f"{stem}.pdf"
    png_path = folder / f"{stem}.png"

    # Keep all explanatory text readable after insertion into a PDF.
    # This also covers manually specified notes, panel labels, cube labels,
    # tick labels, legends, annotations, and colour-bar text.
    for text in fig.findobj(match=Text):
        if text.get_fontsize() < 11:
            text.set_fontsize(11)

    # PDF text and ordinary plot elements remain vector based. The explicit
    # DPI also improves rasterized 3D point fields embedded in the PDF.
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.12,
        dpi=600,
    )
    fig.savefig(
        png_path,
        bbox_inches="tight",
        pad_inches=0.12,
        dpi=600,
    )

    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

def apply_grid(
    ax: plt.Axes,
    axis: str = "y",
) -> None:
    """Apply light journal-style gridlines."""
    ax.grid(
        axis=axis,
        color=COLORS["grid"],
        linewidth=0.65,
        alpha=0.95,
    )
    ax.set_axisbelow(True)

def panel_label(
    ax: plt.Axes,
    text: str,
) -> None:
    """Add a panel heading."""
    ax.text(
        0.0,
        1.04,
        text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )

def normalize_column_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize common ASCII/Unicode column naming variants."""
    rename_map = {
        "R2": "R²",
        "XS_R2": "XS_R²",
        "Alpha_annualized": "Alpha_Annualized",
        "Alpha_annual": "Alpha_Annualized",
        "Test_Monthly_Sharpe": "Monthly_Sharpe",
        "Test_Annualized_Sharpe": "Annualized_Sharpe",
    }

    available = {
        old: new
        for old, new in rename_map.items()
        if old in df.columns and new not in df.columns
    }

    return df.rename(columns=available)

def clean_section_name(
    name: str,
) -> str:
    """Strip AP-Tree weight suffixes and TripleSort suffixes."""
    value = str(name)

    value = re.sub(
        r"_dw_power\d+(?:\.\d+)?$",
        "",
        value,
    )

    value = re.sub(
        r"_triplesort$",
        "",
        value,
    )

    return value

def section_number(
    section: str,
) -> int | None:
    """Extract number from Sec01, Sec02, etc."""
    match = re.search(
        r"^Sec(\d+)",
        str(section),
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return int(match.group(1))

def parse_feature_sequence(
    text: str,
) -> list[str]:
    """
    Parse consecutive raw feature names.

    Example:
        mkt_cap_idio_vol_mkt_cap_st_rev
    """
    remaining = str(text).strip("_")
    output = []

    candidates = sorted(
        RAW_FEATURES,
        key=len,
        reverse=True,
    )

    while remaining:
        matched_feature = None

        for feature in candidates:
            if remaining == feature:
                matched_feature = feature
                remaining = ""
                break

            if remaining.startswith(f"{feature}_"):
                matched_feature = feature
                remaining = remaining[len(feature) + 1:]
                break

        if matched_feature is None:
            return []

        output.append(matched_feature)

    return output

def section_raw_features(
    section: str,
) -> list[str]:
    """Parse the three characteristics in a Section name."""
    section = clean_section_name(section)

    stripped = re.sub(
        r"^Sec\d+_?",
        "",
        section,
        flags=re.IGNORECASE,
    )

    return parse_feature_sequence(stripped)

def feature_display_name(
    feature: str,
) -> str:
    """Convert raw feature name to display name."""
    return FEATURE_MAP.get(feature, feature)

def feature_combination(
    section: str,
) -> str:
    """Return readable feature combination."""
    features = section_raw_features(section)

    if not features:
        return clean_section_name(section)

    return " × ".join(
        feature_display_name(feature)
        for feature in features
    )

def build_section_map(
    sections: Iterable[str],
) -> pd.DataFrame:
    """Build mapping from internal section name to sec1, sec2, etc."""
    unique_sections = sorted({
        clean_section_name(section)
        for section in sections
        if pd.notna(section)
    })

    rows = []
    used_numbers: set[int] = set()

    for section in unique_sections:
        number = section_number(section)

        if number is not None:
            used_numbers.add(number)

    next_number = 1

    for section in unique_sections:
        number = section_number(section)

        if number is None:
            while next_number in used_numbers:
                next_number += 1

            number = next_number
            used_numbers.add(number)
            next_number += 1

        rows.append({
            "Section_Internal_Name": section,
            "Section_Display_Name": f"sec{number}",
            "Section_Number": number,
            "Feature_Combination": feature_combination(section),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("Section_Number")
        .reset_index(drop=True)
    )

def section_display_name(
    section: str,
    section_map: pd.DataFrame,
) -> str:
    """Return secN label."""
    section = clean_section_name(section)

    found = section_map.loc[
        section_map["Section_Internal_Name"].eq(section),
        "Section_Display_Name",
    ]

    if not found.empty:
        return str(found.iloc[0])

    number = section_number(section)

    if number is not None:
        return f"sec{number}"

    return section.lower()

def canonical_strategy_key(
    strategy: str,
    weight_scheme: str,
) -> str:
    """
    Create sample-independent strategy labels.

    Full:
        TripleSort64, APTree_K20, APTree_K40

    Cleaned:
        TripleSort64_clean, APTree_clean_K20, APTree_clean_K40

    are mapped into the same five plotted lines.
    """
    strategy = str(strategy)
    weight_scheme = str(weight_scheme)

    if (
        strategy.startswith("TripleSort64")
        or strategy.startswith("Triple64")
    ):
        return "TripleSort64"

    if (
        "APTree_K20" in strategy
        or "APTree_clean_K20" in strategy
        or "AP20" in strategy
    ):
        return f"APTree_K20_{weight_scheme}"

    if (
        "APTree_K40" in strategy
        or "APTree_clean_K40" in strategy
        or "AP40" in strategy
    ):
        return f"APTree_K40_{weight_scheme}"

    return f"{strategy}_{weight_scheme}"

def strategy_label(
    strategy: str,
) -> str:
    """Return readable strategy label."""
    return STRATEGY_LABELS.get(strategy, strategy)

def strategy_color(
    strategy: str,
) -> str:
    """Return the fixed colour for each plotted strategy."""
    if strategy == "TripleSort64":
        return COLORS["triple"]

    if "K20_dw_power0.5" in strategy:
        return COLORS["ap20_deep"]

    if "K20_dw_power2.0" in strategy:
        return COLORS["ap20_shallow"]

    if "K40_dw_power0.5" in strategy:
        return COLORS["ap40_deep"]

    if "K40_dw_power2.0" in strategy:
        return COLORS["ap40_shallow"]

    return COLORS["neutral"]

def strategy_marker(
    strategy: str,
) -> str:
    """Return the fixed marker for each strategy family."""
    if strategy == "TripleSort64":
        return "o"

    if "dw_power0.5" in strategy:
        return "v"

    if "dw_power2.0" in strategy:
        return "^"

    return "o"

def depth_weight_label(
    weight_scheme: str,
) -> str:
    """
    Return the only two permitted AP-Tree depth-weight descriptions.

    dw_power0.5 is always displayed as Deep Boost.
    dw_power2.0 is always displayed as Shallow Tilt.
    """
    value = str(weight_scheme)

    if value == "dw_power0.5":
        return "Deep Boost"

    if value == "dw_power2.0":
        return "Shallow Tilt"

    return ""

def cube_strategy_description(
    strategy: str,
    model_name: str,
    depth_strategy: str,
) -> str:
    """
    Return a publication-ready cube strategy description.

    AP-Tree depth-weight terminology is restricted to Deep Boost or
    Shallow Tilt. TripleSort64 does not receive a depth-weight label.
    """
    strategy = str(strategy)
    model_name = str(model_name)

    if "APTree" in strategy:
        depth_label = depth_weight_label(depth_strategy)

        if depth_label:
            return f"{model_name}; {depth_label}"

        return model_name

    return model_name

def present_strategies(
    observed: Iterable[str],
) -> list[str]:
    """Return observed strategy keys in fixed publication order."""
    observed = set(observed)

    return [
        strategy
        for strategy in STRATEGY_ORDER
        if strategy in observed
    ]

def make_strategy_legend(
    strategies: list[str],
) -> list[plt.Line2D]:
    """Build reusable legend handles with strategy-specific markers."""
    return [
        plt.Line2D(
            [0],
            [0],
            color=strategy_color(strategy),
            marker=strategy_marker(strategy),
            linewidth=2.2,
            markersize=7.5,
            markerfacecolor=strategy_color(strategy),
            markeredgecolor="#4A4A4A",
            markeredgewidth=0.55,
            label=strategy_label(strategy).replace("\n", " "),
        )
        for strategy in strategies
    ]

# =============================================================================
# Input loaders
# =============================================================================

def load_regression() -> pd.DataFrame | None:
    """Load Step-4 long-format regression output."""
    df = read_csv(PATHS["regression"])

    if df is None or df.empty:
        return None

    df = normalize_column_names(df).copy()

    required = {
        "Sample",
        "Strategy",
        "Weight_Scheme",
        "Section",
        "Model",
        "Alpha_Annualized",
        "R²",
        "Monthly_Sharpe",
    }

    missing = required.difference(df.columns)

    if missing:
        print(
            "SKIP regression figures; missing columns: "
            f"{sorted(missing)}"
        )
        return None

    df["Weight_Scheme"] = (
        df["Weight_Scheme"]
        .fillna("none")
        .astype(str)
    )

    df["Strategy_Key"] = [
        canonical_strategy_key(strategy, scheme)
        for strategy, scheme in zip(
            df["Strategy"],
            df["Weight_Scheme"],
        )
    ]

    numeric_columns = [
        "Alpha_Annualized",
        "R²",
        "XS_R²",
        "Monthly_Sharpe",
        "Annualized_Sharpe",
        "Max_Drawdown",
        "Monthly_Avg_Excess_Return",
        "Selected_Nodes",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df["Section"] = df["Section"].map(clean_section_name)

    return df

def load_xs_r2() -> pd.DataFrame | None:
    """Load cleaned-sample XS-R² output."""
    df = read_csv(PATHS["xs_r2"])

    if df is None or df.empty:
        return None

    df = normalize_column_names(df).copy()

    required = {
        "Sample",
        "Strategy",
        "Weight_Scheme",
        "Section",
        "Model",
        "Status",
        "XS_R²",
    }

    missing = required.difference(df.columns)

    if missing:
        print(
            "SKIP XS-R² merge; missing columns: "
            f"{sorted(missing)}"
        )
        return None

    df = df.loc[
        df["Status"].astype(str).eq("ok")
    ].copy()

    df["Weight_Scheme"] = (
        df["Weight_Scheme"]
        .fillna("none")
        .astype(str)
    )

    df["Strategy_Key"] = [
        canonical_strategy_key(strategy, scheme)
        for strategy, scheme in zip(
            df["Strategy"],
            df["Weight_Scheme"],
        )
    ]

    df["Section"] = df["Section"].map(clean_section_name)
    df["XS_R²"] = pd.to_numeric(
        df["XS_R²"],
        errors="coerce",
    )

    return df

def load_turnover() -> pd.DataFrame | None:
    """Load turnover summary."""
    df = read_csv(PATHS["turnover"])

    if df is None or df.empty:
        return None

    required = {
        "Model",
        "Weight_Scheme",
        "Average_Monthly_Turnover",
    }

    missing = required.difference(df.columns)

    if missing:
        print(
            "SKIP turnover charts; missing columns: "
            f"{sorted(missing)}"
        )
        return None

    df = df.copy()

    df["Weight_Scheme"] = (
        df["Weight_Scheme"]
        .fillna("none")
        .astype(str)
    )

    df["Strategy_Key"] = [
        canonical_strategy_key(model, scheme)
        for model, scheme in zip(
            df["Model"],
            df["Weight_Scheme"],
        )
    ]

    df["Turnover"] = pd.to_numeric(
        df["Average_Monthly_Turnover"],
        errors="coerce",
    )

    section_column = None

    for candidate in [
        "Section",
        "Section_SDF",
        "Section_Name",
    ]:
        if candidate in df.columns:
            section_column = candidate
            break

    if section_column is not None:
        df["Section"] = (
            df[section_column]
            .map(clean_section_name)
        )

    return df

def load_weight_distribution() -> pd.DataFrame | None:
    """Load Top_SDF_Weight_Distributions.csv."""
    df = read_csv(PATHS["weight_distribution"])

    if df is None or df.empty:
        return None

    required = {
        "Sample",
        "Rank",
        "Strategy",
        "Model_Name",
        "Depth_Strategy",
        "Section",
        "Section_SDF",
        "Node_Name",
        "Weight",
    }

    missing = required.difference(df.columns)

    if missing:
        print(
            "SKIP cube charts; missing columns: "
            f"{sorted(missing)}"
        )
        return None

    df = df.copy()

    numeric_columns = [
        "Rank",
        "Weight",
        "Monthly_Sharpe",
        "Annualized_Sharpe",
        "Selected_Nodes",
        "Best_Lambda0",
        "Best_Lambda2",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df["Section"] = df["Section"].map(clean_section_name)

    return df

# =============================================================================
# Merge and summary preparation
# =============================================================================

def merge_xs_r2(
    regression: pd.DataFrame | None,
    xs_r2: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Fill missing regression XS-R² values from detailed cleaned results."""
    if regression is None:
        return None

    reg = regression.copy()

    if "XS_R²" not in reg.columns:
        reg["XS_R²"] = np.nan

    if xs_r2 is None or xs_r2.empty:
        return reg

    xs = xs_r2[
        [
            "Sample",
            "Strategy_Key",
            "Section",
            "Model",
            "XS_R²",
        ]
    ].drop_duplicates(
        subset=[
            "Sample",
            "Strategy_Key",
            "Section",
            "Model",
        ]
    )

    xs = xs.rename(
        columns={"XS_R²": "XS_R²_detail"}
    )

    reg = reg.merge(
        xs,
        on=[
            "Sample",
            "Strategy_Key",
            "Section",
            "Model",
        ],
        how="left",
    )

    reg["XS_R²"] = reg["XS_R²"].fillna(
        reg["XS_R²_detail"]
    )

    return reg.drop(columns=["XS_R²_detail"])

def build_summary(
    regression: pd.DataFrame | None,
    sample: str,
) -> pd.DataFrame | None:
    """Collapse repeated factor-model rows into Strategy × Section rows."""
    if regression is None or regression.empty:
        return None

    subset = regression.loc[
        regression["Sample"].astype(str).eq(sample)
    ].copy()

    if subset.empty:
        return None

    columns = [
        "Section",
        "Strategy_Key",
        "Monthly_Sharpe",
        "Annualized_Sharpe",
        "Monthly_Avg_Excess_Return",
        "Max_Drawdown",
        "Selected_Nodes",
    ]

    columns = [
        column
        for column in columns
        if column in subset.columns
    ]

    return (
        subset[columns]
        .drop_duplicates(
            subset=[
                "Section",
                "Strategy_Key",
            ],
            keep="first",
        )
        .copy()
    )

def pivot_section_metric(
    data: pd.DataFrame,
    metric: str,
    sample: str,
    factor_model: str | None = None,
) -> pd.DataFrame | None:
    """
    Pivot one metric to Section × Strategy.

    Sorting is always based on TripleSort64 ascending values.
    """
    subset = data.loc[
        data["Sample"].astype(str).eq(sample)
        & data["Strategy_Key"].isin(STRATEGY_ORDER)
    ].copy()

    if factor_model is not None:
        subset = subset.loc[
            subset["Model"].astype(str).eq(factor_model)
        ]

    if subset.empty or metric not in subset.columns:
        return None

    subset = subset.drop_duplicates(
        subset=[
            "Section",
            "Strategy_Key",
        ],
        keep="first",
    )

    wide = subset.pivot(
        index="Section",
        columns="Strategy_Key",
        values=metric,
    )

    if wide.empty:
        return None

    strategies = present_strategies(
        wide.columns
    )

    if len(strategies) < 2:
        return None

    wide = wide.reindex(columns=strategies)

    if "TripleSort64" in wide.columns:
        wide = wide.loc[
            wide["TripleSort64"].sort_values(
                na_position="last"
            ).index
        ]
    else:
        wide = wide.sort_index()

    return wide

# =============================================================================
# Appendix boxplots
# =============================================================================

def plot_overall_boxplots(
    cleaned_summary: pd.DataFrame | None,
) -> None:
    """Appendix performance distribution boxplots."""
    if cleaned_summary is None or cleaned_summary.empty:
        return

    strategies = present_strategies(
        cleaned_summary["Strategy_Key"].unique()
    )

    if not strategies:
        return

    specs = [
        (
            "Monthly_Sharpe",
            "Monthly out-of-sample Sharpe ratio",
            "Panel A — Sharpe Ratio",
            False,
        ),
        (
            "Monthly_Avg_Excess_Return",
            "Monthly mean excess return",
            "Panel B — Average Excess Return",
            True,
        ),
        (
            "Max_Drawdown",
            "Maximum drawdown",
            "Panel C — Maximum Drawdown",
            True,
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.8, 5.8),
    )

    for ax, (
        metric,
        xlabel,
        title,
        percent,
    ) in zip(axes, specs):
        values = [
            cleaned_summary.loc[
                cleaned_summary["Strategy_Key"].eq(strategy),
                metric,
            ].dropna()
            for strategy in strategies
        ]

        boxes = ax.boxplot(
            values,
            vert=False,
            patch_artist=True,
            widths=0.58,
            medianprops={
                "color": "black",
                "linewidth": 1.1,
            },
        )

        for box, strategy in zip(
            boxes["boxes"],
            strategies,
        ):
            box.set_facecolor(
                strategy_color(strategy)
            )
            box.set_edgecolor("none")
            box.set_alpha(0.88)

        ax.axvline(
            0.0,
            color="black",
            linewidth=0.70,
        )

        ax.set_yticks(range(1, len(strategies) + 1))
        ax.set_yticklabels([
            strategy_label(strategy).replace("\n", " ")
            for strategy in strategies
        ])

        ax.set_xlabel(xlabel)
        ax.set_title(title, pad=13)

        if percent:
            ax.xaxis.set_major_formatter(
                mtick.PercentFormatter(xmax=1.0)
            )

        apply_grid(ax, axis="x")

    fig.suptitle(
        "Appendix: Cleaned-Sample Section-SDF Performance Distributions",
        fontsize=12,
        y=1.02,
    )

    fig.tight_layout()

    save(
        fig,
        APPENDIX_DIR,
        "A01_Overall_Performance_Boxplots",
    )

def plot_turnover_boxplot(
    turnover: pd.DataFrame | None,
) -> None:
    """Appendix turnover distribution boxplot."""
    if turnover is None or turnover.empty:
        return

    strategies = present_strategies(
        turnover["Strategy_Key"].unique()
    )

    if not strategies:
        return

    fig, ax = plt.subplots(
        figsize=(10.8, 5.8),
    )

    values = [
        turnover.loc[
            turnover["Strategy_Key"].eq(strategy),
            "Turnover",
        ].dropna()
        for strategy in strategies
    ]

    boxes = ax.boxplot(
        values,
        vert=False,
        patch_artist=True,
        widths=0.58,
        medianprops={
            "color": "black",
            "linewidth": 1.1,
        },
    )

    for box, strategy in zip(
        boxes["boxes"],
        strategies,
    ):
        box.set_facecolor(
            strategy_color(strategy)
        )
        box.set_edgecolor("none")
        box.set_alpha(0.88)

    ax.set_yticks(range(1, len(strategies) + 1))
    ax.set_yticklabels([
        strategy_label(strategy).replace("\n", " ")
        for strategy in strategies
    ])

    ax.set_xlabel(
        "Average monthly one-way turnover"
    )

    ax.xaxis.set_major_formatter(
        mtick.PercentFormatter(xmax=1.0)
    )

    ax.set_title(
        "Appendix: Cleaned-Sample Turnover Distribution",
        pad=13,
    )

    apply_grid(ax, axis="x")

    fig.tight_layout()

    save(
        fig,
        APPENDIX_DIR,
        "A05_Turnover_Distribution_Boxplot",
    )

# =============================================================================
# Deep Boost versus Shallow Tilt comparison
# =============================================================================

def paired_depth_difference(
    cleaned_summary: pd.DataFrame,
    k_value: int,
) -> pd.DataFrame | None:
    """Compute Deep Boost minus Shallow Tilt differences."""
    deep = f"APTree_K{k_value}_dw_power0.5"
    shallow = f"APTree_K{k_value}_dw_power2.0"

    subset = cleaned_summary.loc[
        cleaned_summary["Strategy_Key"].isin([
            deep,
            shallow,
        ])
    ].copy()

    if subset.empty:
        return None

    wide = subset.pivot(
        index="Section",
        columns="Strategy_Key",
        values=[
            "Monthly_Sharpe",
            "Monthly_Avg_Excess_Return",
            "Max_Drawdown",
        ],
    )

    if (
        deep not in wide["Monthly_Sharpe"].columns
        or shallow not in wide["Monthly_Sharpe"].columns
    ):
        return None

    output = pd.DataFrame(index=wide.index)

    output["Delta_Sharpe"] = (
        wide["Monthly_Sharpe"][deep]
        - wide["Monthly_Sharpe"][shallow]
    )

    output["Delta_Return"] = (
        wide["Monthly_Avg_Excess_Return"][deep]
        - wide["Monthly_Avg_Excess_Return"][shallow]
    )

    output["Delta_Drawdown"] = (
        wide["Max_Drawdown"][deep]
        - wide["Max_Drawdown"][shallow]
    )

    return output.dropna(how="all")

def plot_depth_weight_performance(
    cleaned_summary: pd.DataFrame | None,
    section_map: pd.DataFrame,
) -> None:
    """Plot Deep Boost minus Shallow Tilt performance differences."""
    if cleaned_summary is None or cleaned_summary.empty:
        return

    specs = [
        (
            "Delta_Sharpe",
            "Monthly Sharpe difference",
            False,
        ),
        (
            "Delta_Return",
            "Monthly mean excess-return difference",
            True,
        ),
        (
            "Delta_Drawdown",
            "Maximum drawdown difference",
            True,
        ),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17.2, 12.4),
        sharey="row",
    )

    letters = iter("ABCDEF")

    for row_index, k_value in enumerate([20, 40]):
        data = paired_depth_difference(
            cleaned_summary,
            k_value,
        )

        if data is None or data.empty:
            continue

        data = data.sort_values("Delta_Sharpe")
        y = np.arange(len(data))

        for column_index, (
            metric,
            xlabel,
            percent,
        ) in enumerate(specs):
            ax = axes[row_index, column_index]
            values = data[metric]

            colours = np.where(
                values >= 0.0,
                COLORS["positive"],
                COLORS["negative"],
            )

            ax.barh(
                y,
                values,
                color=colours,
                height=0.67,
                alpha=0.92,
            )

            ax.axvline(
                0.0,
                color="black",
                linewidth=0.78,
            )

            ax.set_xlabel(xlabel)

            if percent:
                ax.xaxis.set_major_formatter(
                    mtick.PercentFormatter(xmax=1.0)
                )

            if column_index == 0:
                ax.set_yticks(y)
                ax.set_yticklabels([
                    section_display_name(
                        section,
                        section_map,
                    )
                    for section in data.index
                ])
                ax.set_ylabel(
                    f"K = {k_value}: Sections"
                )
            else:
                ax.tick_params(
                    axis="y",
                    left=False,
                    labelleft=False,
                )

            wins = int((values > 0.0).sum())

            ax.set_title(
                f"K = {k_value}; Deep Boost wins "
                f"{wins}/{len(data)} sections",
                pad=13,
            )

            panel_label(
                ax,
                f"Panel {next(letters)}",
            )

            apply_grid(ax, axis="x")

    fig.suptitle(
        "Deep Boost minus Shallow Tilt AP-Tree Performance",
        fontsize=12,
        y=1.01,
    )

    fig.tight_layout()

    save(
        fig,
        MAIN_DIR,
        "02_Depth_Weight_Performance_Differences",
    )

# =============================================================================
# Line-chart utilities
# =============================================================================

def draw_section_lines(
    ax: plt.Axes,
    wide: pd.DataFrame,
    section_map: pd.DataFrame,
    ylabel: str,
    percent: bool = False,
    title: str | None = None,
    show_legend: bool = False,
) -> None:
    """
    Draw model lines over sections using fixed colours and markers.

    Rows of wide are already sorted by TripleSort64 ascending values.
    """
    strategies = present_strategies(
        wide.columns
    )

    x = np.arange(len(wide))

    for strategy in strategies:
        ax.plot(
            x,
            wide[strategy].to_numpy(dtype=float),
            color=strategy_color(strategy),
            linewidth=2.2,
            marker=strategy_marker(strategy),
            markersize=6.5,
            markerfacecolor=strategy_color(strategy),
            markeredgecolor="#4A4A4A",
            markeredgewidth=0.55,
            label=strategy_label(strategy).replace("\n", " "),
            zorder=3,
        )

    ax.axhline(
        0.0,
        color="black",
        linewidth=0.85,
        zorder=1,
    )

    ax.set_xticks(x)

    ax.set_xticklabels([
        section_display_name(
            section,
            section_map,
        )
        for section in wide.index
    ], rotation=90)

    ax.set_xlim(
        -0.50,
        len(wide) - 0.50,
    )

    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title, pad=14)

    if percent:
        ax.yaxis.set_major_formatter(
            mtick.PercentFormatter(xmax=1.0)
        )

    if show_legend:
        ax.legend(
            handles=make_strategy_legend(strategies),
            frameon=False,
            ncol=3,
            loc="upper left",
        )

    apply_grid(ax, axis="y")

# =============================================================================
# Full and Cleaned Sharpe line charts
# =============================================================================

def plot_sharpe_by_section(
    regression: pd.DataFrame | None,
    sample: str,
    section_map: pd.DataFrame,
) -> None:
    """
    Plot Full or Cleaned monthly Sharpe ratios.

    Dimensions intentionally match about one third of the total height of
    the three-panel factor-model figures.
    """
    if regression is None or regression.empty:
        return

    wide = pivot_section_metric(
        data=regression,
        metric="Monthly_Sharpe",
        sample=sample,
    )

    if wide is None or wide.empty:
        print(
            f"SKIP {sample} Sharpe line chart: no usable rows."
        )
        return

    fig, ax = plt.subplots(
        figsize=(16.4, 8.6),
    )

    draw_section_lines(
        ax=ax,
        wide=wide,
        section_map=section_map,
        ylabel="Monthly out-of-sample Sharpe ratio",
        title=(
            f"{sample} Sample: Monthly Out-of-Sample Sharpe Ratio by Section"
        ),
        show_legend=True,
    )

    ax.set_xlabel(
        "Sections sorted by TripleSort64 monthly Sharpe ratio in ascending order"
    )

    fig.tight_layout()

    save(
        fig,
        MAIN_DIR,
        f"03_{sample}_Sharpe_By_Section",
    )

# =============================================================================
# Factor-model evidence charts
# =============================================================================

def plot_factor_evidence_by_section(
    regression: pd.DataFrame | None,
    factor_model: str,
    section_map: pd.DataFrame,
    output_folder: Path,
    output_stem: str,
) -> None:
    """Plot Alpha, R², and XS-R² for one factor model."""
    if regression is None or regression.empty:
        return

    metric_specs = [
        (
            "Alpha_Annualized",
            "Annualized alpha",
            True,
            "Panel A — Annualized Alpha",
        ),
        (
            "R²",
            "Time-series $R^2$",
            True,
            "Panel B — Time-Series $R^2$",
        ),
        (
            "XS_R²",
            "Selected-node XS-$R^2$",
            True,
            "Panel C — Cross-Sectional XS-$R^2$",
        ),
    ]

    wide_frames = []

    for metric, _, _, _ in metric_specs:
        wide = pivot_section_metric(
            data=regression,
            metric=metric,
            sample="Cleaned",
            factor_model=factor_model,
        )

        if wide is None or wide.empty:
            print(
                f"SKIP {factor_model}: no usable {metric} data."
            )
            return

        wide_frames.append(wide)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16.4, 20.0),
        sharex=False,
    )

    for ax, wide, (
        _,
        ylabel,
        percent,
        title,
    ) in zip(
        axes,
        wide_frames,
        metric_specs,
    ):
        draw_section_lines(
            ax=ax,
            wide=wide,
            section_map=section_map,
            ylabel=ylabel,
            percent=percent,
            title=title,
        )

    strategies = present_strategies(
        wide_frames[0].columns
    )

    axes[0].legend(
        handles=make_strategy_legend(strategies),
        frameon=False,
        ncol=3,
        loc="upper left",
    )

    axes[-1].set_xlabel(
        "Sections sorted separately by TripleSort64 values in ascending order"
    )

    fig.suptitle(
        "Cleaned Sample: Section-Level Factor Evidence under "
        f"the {MODEL_LABELS[factor_model]}",
        fontsize=12,
        y=0.996,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.985),
    )

    save(
        fig,
        output_folder,
        output_stem,
    )

# =============================================================================
# Turnover chart
# =============================================================================

def plot_turnover_by_section(
    turnover: pd.DataFrame | None,
    section_map: pd.DataFrame,
) -> None:
    """Plot cleaned-sample turnover across sections and strategies."""
    if turnover is None or turnover.empty:
        return

    if "Section" not in turnover.columns:
        print(
            "SKIP turnover line chart: no Section column."
        )
        return

    subset = turnover.loc[
        turnover["Strategy_Key"].isin(
            STRATEGY_ORDER
        )
    ].copy()

    if subset.empty:
        return

    wide = subset.pivot_table(
        index="Section",
        columns="Strategy_Key",
        values="Turnover",
        aggfunc="mean",
    )

    if wide.empty:
        return

    strategies = present_strategies(
        wide.columns
    )

    if len(strategies) < 2:
        return

    wide = wide.reindex(columns=strategies)

    if "TripleSort64" in wide.columns:
        wide = wide.loc[
            wide["TripleSort64"].sort_values(
                na_position="last"
            ).index
        ]

    fig, ax = plt.subplots(
        figsize=(16.4, 8.6),
    )

    draw_section_lines(
        ax=ax,
        wide=wide,
        section_map=section_map,
        ylabel="Average monthly one-way turnover",
        percent=True,
        title=(
            "Cleaned Sample: Average Monthly Turnover by Section"
        ),
        show_legend=True,
    )

    ax.set_xlabel(
        "Sections sorted by TripleSort64 turnover in ascending order"
    )

    fig.tight_layout()

    save(
        fig,
        MAIN_DIR,
        "05_Turnover_By_Section",
    )

# =============================================================================
# Main combined factor chart
# =============================================================================

def plot_main_factor_evidence_combined(
    regression: pd.DataFrame | None,
    section_map: pd.DataFrame,
) -> None:
    """Create 3 × 3 figure for CH4, FF5, Q5 and Alpha/R²/XS-R²."""
    if regression is None or regression.empty:
        return

    metric_specs = [
        (
            "Alpha_Annualized",
            "Annualized alpha",
            True,
            "Annualized Alpha",
        ),
        (
            "R²",
            "Time-series $R^2$",
            True,
            "Time-Series $R^2$",
        ),
        (
            "XS_R²",
            "Selected-node XS-$R^2$",
            True,
            "Cross-Sectional XS-$R^2$",
        ),
    ]

    data: dict[tuple[str, str], pd.DataFrame] = {}

    for factor_model in MAIN_FACTOR_MODELS:
        for metric, _, _, _ in metric_specs:
            wide = pivot_section_metric(
                data=regression,
                metric=metric,
                sample="Cleaned",
                factor_model=factor_model,
            )

            if wide is None or wide.empty:
                print(
                    f"SKIP combined factor chart: "
                    f"{factor_model} / {metric} unavailable."
                )
                return

            data[(factor_model, metric)] = wide

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(23.0, 19.5),
    )

    for row, factor_model in enumerate(MAIN_FACTOR_MODELS):
        for col, (
            metric,
            ylabel,
            percent,
            column_title,
        ) in enumerate(metric_specs):
            ax = axes[row, col]

            draw_section_lines(
                ax=ax,
                wide=data[(factor_model, metric)],
                section_map=section_map,
                ylabel=ylabel,
                percent=percent,
                title=column_title if row == 0 else None,
            )

            if col == 0:
                ax.text(
                    -0.17,
                    0.50,
                    MODEL_LABELS[factor_model],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )

            if row == 2:
                ax.set_xlabel(
                    "Sections sorted by TripleSort64"
                )

    first_frame = data[
        (MAIN_FACTOR_MODELS[0], metric_specs[0][0])
    ]

    fig.legend(
        handles=make_strategy_legend(
            present_strategies(first_frame.columns)
        ),
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )

    fig.suptitle(
        "Cleaned Sample: Main Factor-Model Evidence across Section-SDFs",
        fontsize=14,
        y=1.015,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.965),
    )

    save(
        fig,
        MAIN_DIR,
        "11_Main_Factor_Evidence_Combined",
    )

def plot_full_cleaned_sharpe_combined(
    regression: pd.DataFrame | None,
    section_map: pd.DataFrame,
) -> None:
    """Combine Full and Cleaned Sharpe line charts."""
    if regression is None or regression.empty:
        return

    full = pivot_section_metric(
        data=regression,
        metric="Monthly_Sharpe",
        sample="Full",
    )

    cleaned = pivot_section_metric(
        data=regression,
        metric="Monthly_Sharpe",
        sample="Cleaned",
    )

    if (
        full is None
        or cleaned is None
        or full.empty
        or cleaned.empty
    ):
        print(
            "SKIP Full/Cleaned Sharpe combined chart."
        )
        return

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(17.2, 15.0),
    )

    draw_section_lines(
        ax=axes[0],
        wide=full,
        section_map=section_map,
        ylabel="Monthly out-of-sample Sharpe ratio",
        title="Panel A — Full Sample",
        show_legend=True,
    )

    draw_section_lines(
        ax=axes[1],
        wide=cleaned,
        section_map=section_map,
        ylabel="Monthly out-of-sample Sharpe ratio",
        title="Panel B — Cleaned Sample",
    )

    axes[0].set_xlabel(
        "Sections sorted by Full-sample TripleSort64 Sharpe"
    )

    axes[1].set_xlabel(
        "Sections sorted by Cleaned-sample TripleSort64 Sharpe"
    )

    fig.suptitle(
        "Monthly Out-of-Sample Sharpe Ratios: Full versus Cleaned Samples",
        fontsize=13,
        y=0.998,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.985),
    )

    save(
        fig,
        MAIN_DIR,
        "12_Full_Cleaned_Sharpe_Combined",
    )

# =============================================================================
# 3D cube node parsing
# =============================================================================

def parse_ap_tree_node_path(
    node_name: str,
) -> tuple[list[str], list[int]] | None:
    """
    Parse an AP-Tree root, intermediate node, or leaf node.

    Node-name format:
        <section>-<complete split-feature sequence>[_<direction path>]

    The feature sequence after '-' may contain features that have not yet
    been used. The number of trailing direction digits determines the actual
    node depth, so only the first ``depth`` features are applied.

    Examples:
        Sec11_mkt_cap_st_rev_idio_vol-
        idio_vol_idio_vol_idio_vol_idio_vol
            -> ([], [])

        Sec11_mkt_cap_st_rev_idio_vol-
        idio_vol_st_rev_st_rev_idio_vol_111
            -> ([idio_vol, st_rev, st_rev], [1, 1, 1])

        Sec11_mkt_cap_st_rev_idio_vol-
        mkt_cap_mkt_cap_idio_vol_idio_vol_11
            -> ([mkt_cap, mkt_cap], [1, 1])

        Sec09_mkt_cap_st_rev_r12_2-
        r12_2_mkt_cap_mkt_cap_mkt_cap_1
            -> ([r12_2], [1])

    Parsing the complete text as a feature sequence first is intentional.
    It prevents a final feature such as ``r12_2`` from being mistaken for a
    direction suffix in a root-node name.
    """
    text = str(node_name).strip()

    if "-" not in text:
        return None

    _, path = text.rsplit("-", maxsplit=1)
    path = path.strip().strip("_")

    if not path:
        return None

    # A path made entirely of feature names is the root/full-sample node.
    complete_features = parse_feature_sequence(path)

    if complete_features:
        return [], []

    # Non-root nodes end in a direction path consisting of 1 and 2.
    matched = re.search(r"_([12]+)$", path)

    if matched is None:
        return None

    direction_part = matched.group(1)
    feature_part = path[:matched.start()]

    all_features = parse_feature_sequence(feature_part)

    if not all_features:
        return None

    directions = [int(value) for value in direction_part]
    depth = len(directions)

    if depth > len(all_features):
        return None

    # Only the first `depth` features have actually been used for splitting.
    used_features = all_features[:depth]

    return used_features, directions

def ap_tree_bounds(
    node_name: str,
    section_features: list[str],
) -> dict[str, tuple[float, float]] | None:
    """Translate conditional AP-Tree splits to continuous intervals in [0, 1]."""
    parsed = parse_ap_tree_node_path(node_name)

    if parsed is None:
        return None

    split_features, directions = parsed

    bounds = {
        feature: (0.0, 1.0)
        for feature in section_features
    }

    for feature, direction in zip(
        split_features,
        directions,
    ):
        if feature not in bounds:
            continue

        lower, upper = bounds[feature]
        midpoint = (lower + upper) / 2.0

        if direction == 1:
            bounds[feature] = (lower, midpoint)
        else:
            bounds[feature] = (midpoint, upper)

    return bounds

def parse_triplesort64_node(
    node_name: str,
) -> list[int] | None:
    """
    Parse TripleSort64 cell names such as 1_3_3.

    1 -> 0%-25%
    2 -> 25%-50%
    3 -> 50%-75%
    4 -> 75%-100%
    """
    digits = re.findall(
        r"[1-4]",
        str(node_name),
    )

    if len(digits) < 3:
        return None

    result = [
        int(value)
        for value in digits[-3:]
    ]

    if not all(
        1 <= value <= 4
        for value in result
    ):
        return None

    return result

def triplesort64_bounds(
    node_name: str,
    section_features: list[str],
) -> dict[str, tuple[float, float]] | None:
    """Map TripleSort64 quartile cell to three feature intervals."""
    digits = parse_triplesort64_node(node_name)

    if digits is None or len(section_features) != 3:
        return None

    bounds = {}

    for feature, digit in zip(
        section_features,
        digits,
    ):
        bounds[feature] = (
            (digit - 1) / 4.0,
            digit / 4.0,
        )

    return bounds

# =============================================================================
# 3D cube drawing: cumulative overlapping-weight field
# =============================================================================

CUBE_GRID_SIZE = 64
CUBE_SMOOTH_SIGMA = 1.25
CUBE_RENDER_STRIDE = 1
CUBE_MIN_ALPHA = 0.025
CUBE_MAX_ALPHA = 0.62

def cube_vertices(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
) -> np.ndarray:
    """Return eight vertices of a cuboid."""
    return np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ])

def cube_faces(
    vertices: np.ndarray,
) -> list[list[np.ndarray]]:
    """Return six cuboid faces."""
    return [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
    ]

def transform_lme_bounds(
    bounds: tuple[float, float],
    sample: str,
) -> tuple[float, float] | None:
    """
    Convert original LME bounds to plotted z-axis bounds.

    Full sample:
        Original LME 0%-100% maps directly to z in [0, 1].

    Cleaned sample:
        Original LME 30%-100% is retained and stretched to z in [0, 1].
    """
    lower, upper = bounds

    if sample == "Full":
        return lower, upper

    cutoff = 0.30

    if upper <= cutoff:
        return None

    lower = max(lower, cutoff)

    return (
        (lower - cutoff) / (1.0 - cutoff),
        (upper - cutoff) / (1.0 - cutoff),
    )

def add_full_sample_shell_plane(
    ax,
) -> None:
    """Add the LME = 30% grey plane to Full sample cubes."""
    z = 0.30

    vertices = np.array([
        [0.0, 0.0, z],
        [1.0, 0.0, z],
        [1.0, 1.0, z],
        [0.0, 1.0, z],
    ])

    plane = Poly3DCollection(
        [[
            vertices[0],
            vertices[1],
            vertices[2],
            vertices[3],
        ]],
        facecolors=[(0.40, 0.40, 0.40, 0.16)],
        edgecolors=[(0.32, 0.32, 0.32, 0.58)],
        linewidths=0.65,
        linestyles="--",
    )

    ax.add_collection3d(plane)

    ax.text(
        0.05,
        0.05,
        0.16,
        "Shell-contamination region\nLME 0-30%",
        fontsize=6.7,
        color="#505050",
    )

def setup_cube_axis(
    ax,
    x_feature: str,
    y_feature: str,
    sample: str,
    title: str | None = None,
) -> None:
    """Configure the three-dimensional coordinate system."""
    ax.set_xlim(1.0, 0.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)

    ax.set_box_aspect((1.0, 1.0, 1.0))

    ax.set_xlabel(
        feature_display_name(x_feature),
        labelpad=8,
    )

    ax.set_ylabel(
        feature_display_name(y_feature),
        labelpad=8,
    )

    ax.set_zlabel(
        "LME",
        labelpad=9,
    )

    ticks = [0.0, 0.25, 0.50, 0.75, 1.0]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.set_xticklabels(
        ["0", "25", "50", "75", "100"],
        fontsize=7,
    )

    ax.set_yticklabels(
        ["0", "25", "50", "75", "100"],
        fontsize=7,
    )

    if sample == "Full":
        ax.set_zticklabels(
            ["0", "25", "50", "75", "100"],
            fontsize=7,
        )
    else:
        ax.set_zticklabels(
            ["30", "47.5", "65", "82.5", "100"],
            fontsize=7,
        )

    ax.view_init(
        elev=22,
        azim=-132,
    )

    if title is not None:
        ax.set_title(
            title,
            pad=16,
            fontsize=9.5,
        )

    for axis in [
        ax.xaxis,
        ax.yaxis,
        ax.zaxis,
    ]:
        axis.pane.fill = False
        axis.pane.set_edgecolor(
            COLORS["cube_grid"]
        )
        axis._axinfo["grid"]["color"] = COLORS["cube_grid"]
        axis._axinfo["grid"]["linewidth"] = 0.38
        axis._axinfo["tick"]["color"] = "#999999"

def bounds_to_grid_slice(
    bounds: tuple[float, float],
    grid_size: int,
) -> slice:
    """Convert a continuous [0, 1] interval to an inclusive grid slice."""
    lower, upper = bounds

    lower = float(np.clip(lower, 0.0, 1.0))
    upper = float(np.clip(upper, 0.0, 1.0))

    start = int(np.floor(lower * grid_size))
    stop = int(np.ceil(upper * grid_size))

    start = max(0, min(grid_size - 1, start))
    stop = max(start + 1, min(grid_size, stop))

    return slice(start, stop)

def build_cumulative_weight_field(
    node_bounds: list[tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        float,
    ]],
    grid_size: int = CUBE_GRID_SIZE,
    smoothing_sigma: float = CUBE_SMOOTH_SIGMA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a three-dimensional cumulative SDF weight field.

    At every grid location, weights of all covering selected nodes are summed.
    Gaussian smoothing affects only rendered colour, not the raw overlap sum.
    """
    raw_weight_field = np.zeros(
        (grid_size, grid_size, grid_size),
        dtype=float,
    )

    coverage_field = np.zeros(
        (grid_size, grid_size, grid_size),
        dtype=np.int16,
    )

    for (
        x_bounds,
        y_bounds,
        z_bounds,
        weight,
    ) in node_bounds:
        x_slice = bounds_to_grid_slice(
            x_bounds,
            grid_size,
        )

        y_slice = bounds_to_grid_slice(
            y_bounds,
            grid_size,
        )

        z_slice = bounds_to_grid_slice(
            z_bounds,
            grid_size,
        )

        raw_weight_field[
            x_slice,
            y_slice,
            z_slice,
        ] += weight

        coverage_field[
            x_slice,
            y_slice,
            z_slice,
        ] += 1

    smooth_weight_field = gaussian_filter(
        raw_weight_field,
        sigma=smoothing_sigma,
        mode="nearest",
    )

    return (
        raw_weight_field,
        smooth_weight_field,
        coverage_field,
    )

def render_cumulative_weight_field(
    ax,
    smooth_weight_field: np.ndarray,
    coverage_field: np.ndarray,
    max_abs_single_node_weight: float,
    cmap,
    norm,
    grid_size: int = CUBE_GRID_SIZE,
) -> None:
    """Render the cumulative weight field as a smooth transparent point volume."""
    occupied = coverage_field > 0

    if not np.any(occupied):
        return

    x_index, y_index, z_index = np.where(occupied)

    values = smooth_weight_field[
        occupied
    ]

    centers_x = (
        x_index + 0.5
    ) / grid_size

    centers_y = (
        y_index + 0.5
    ) / grid_size

    centers_z = (
        z_index + 0.5
    ) / grid_size

    stride = max(1, int(CUBE_RENDER_STRIDE))

    centers_x = centers_x[::stride]
    centers_y = centers_y[::stride]
    centers_z = centers_z[::stride]
    values = values[::stride]

    rgba = cmap(norm(values))

    scaled_magnitude = np.clip(
        np.abs(values) / max_abs_single_node_weight,
        0.0,
        1.0,
    )

    rgba[:, 3] = (
        CUBE_MIN_ALPHA
        + (CUBE_MAX_ALPHA - CUBE_MIN_ALPHA)
        * np.power(scaled_magnitude, 0.52)
    )

    ax.scatter(
        centers_x,
        centers_y,
        centers_z,
        c=rgba,
        s=4.2,
        marker="s",
        linewidths=0.0,
        depthshade=False,
        rasterized=True,
    )

def get_cube_data(
    weight_distribution: pd.DataFrame,
    sample: str,
    rank: int,
) -> tuple[pd.DataFrame, pd.Series] | None:
    """Load all selected-node rows for one top-ranked SDF."""
    sdf = weight_distribution.loc[
        weight_distribution["Sample"].astype(str).eq(sample)
        & weight_distribution["Rank"].eq(rank)
    ].copy()

    sdf["Weight"] = pd.to_numeric(
        sdf["Weight"],
        errors="coerce",
    )

    sdf = sdf.dropna(subset=["Weight"])

    if sdf.empty:
        return None

    return sdf, sdf.iloc[0]

def cube_weight_diagnostics(
    sdf: pd.DataFrame,
    metadata: pd.Series,
) -> dict[str, float | int]:
    """
    Calculate performance and effective-weight sparsity diagnostics.

    Effective nodes are calculated from normalized absolute weights:

        p_i = |w_i| / sum_j |w_j|
        Effective Nodes = 1 / sum_i p_i²

    Sparsity is Effective Nodes / selected-node count. A smaller value means
    the SDF weight is concentrated in fewer effective nodes.
    """
    weights = pd.to_numeric(
        sdf["Weight"],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    selected_nodes = int(len(weights))
    absolute_weight_sum = float(np.abs(weights).sum())

    if absolute_weight_sum > 0.0:
        normalized_absolute_weights = (
            np.abs(weights)
            / absolute_weight_sum
        )

        effective_nodes = float(
            1.0
            / np.square(
                normalized_absolute_weights
            ).sum()
        )
    else:
        effective_nodes = np.nan

    sparsity = (
        effective_nodes / selected_nodes
        if (
            selected_nodes > 0
            and np.isfinite(effective_nodes)
        )
        else np.nan
    )

    monthly_sharpe = pd.to_numeric(
        pd.Series([
            metadata.get(
                "Monthly_Sharpe",
                np.nan,
            )
        ]),
        errors="coerce",
    ).iloc[0]

    return {
        "Monthly_Sharpe": float(monthly_sharpe),
        "Selected_Nodes": selected_nodes,
        "Effective_Nodes": effective_nodes,
        "Sparsity": sparsity,
        "Absolute_Weight_Sum": absolute_weight_sum,
    }

def cube_diagnostic_text(
    diagnostics: dict[str, float | int],
) -> str:
    """Format the Sharpe and sparsity information for cube titles."""
    sharpe = diagnostics["Monthly_Sharpe"]
    selected_nodes = diagnostics["Selected_Nodes"]
    effective_nodes = diagnostics["Effective_Nodes"]
    sparsity = diagnostics["Sparsity"]
    absolute_weight_sum = diagnostics["Absolute_Weight_Sum"]

    return (
        f"SR = {sharpe:.3f}; "
        f"N = {selected_nodes}; "
        f"Eff. N = {effective_nodes:.1f}; "
        f"Sparsity = {sparsity:.1%}\n"
        f"Sum |w| = {absolute_weight_sum:.3f}"
    )

def draw_weight_cube(
    ax,
    sdf: pd.DataFrame,
    metadata: pd.Series,
    sample: str,
    section_map: pd.DataFrame,
    title: str | None = None,
) -> ScalarMappable | None:
    """
    Draw one SDF cube as a cumulative overlapping-weight field.

    All selected nodes are parsed and retained. At every overlapping location,
    selected-node weights are summed before smoothing and rendering.
    """
    section = str(metadata["Section"])
    section_features = section_raw_features(section)

    if len(section_features) != 3:
        print(
            f"SKIP cube: cannot parse three features from {section}"
        )
        return None

    if "mkt_cap" not in section_features:
        print(
            f"SKIP cube: mkt_cap missing from {section}"
        )
        return None

    non_size_features = [
        feature
        for feature in section_features
        if feature != "mkt_cap"
    ]

    x_feature = non_size_features[0]
    y_feature = non_size_features[1]
    z_feature = "mkt_cap"

    weights = sdf["Weight"].to_numpy(
        dtype=float
    )

    max_abs_single_node_weight = np.nanmax(
        np.abs(weights)
    )

    if (
        not np.isfinite(max_abs_single_node_weight)
        or max_abs_single_node_weight <= 0.0
    ):
        max_abs_single_node_weight = 1.0

    cmap = plt.get_cmap("RdYlGn")

    norm = mcolors.TwoSlopeNorm(
        vmin=-max_abs_single_node_weight,
        vcenter=0.0,
        vmax=max_abs_single_node_weight,
    )

    if title is None:
        section_label = section_display_name(
            section,
            section_map,
        )

        rank = int(metadata["Rank"])

        description = cube_strategy_description(
            strategy=str(metadata["Strategy"]),
            model_name=str(metadata["Model_Name"]),
            depth_strategy=str(metadata["Depth_Strategy"]),
        )

        diagnostics = cube_weight_diagnostics(
            sdf=sdf,
            metadata=metadata,
        )

        title = (
            f"Rank {rank}: {sample} | {section_label}\n"
            f"{description}\n"
            f"{cube_diagnostic_text(diagnostics)}"
        )

    setup_cube_axis(
        ax=ax,
        x_feature=x_feature,
        y_feature=y_feature,
        sample=sample,
        title=title,
    )

    parsed_node_bounds = []

    for _, row in sdf.iterrows():
        strategy = str(row["Strategy"])
        node_name = str(row["Node_Name"])
        weight = float(row["Weight"])

        if strategy.startswith("TripleSort64"):
            bounds = triplesort64_bounds(
                node_name=node_name,
                section_features=section_features,
            )
        else:
            bounds = ap_tree_bounds(
                node_name=node_name,
                section_features=section_features,
            )

        if bounds is None:
            print(
                f"SKIP unparsed node: "
                f"sample={sample}, rank={metadata['Rank']}, "
                f"strategy={strategy}, node={node_name}"
            )
            continue

        if not all(
            feature in bounds
            for feature in [
                x_feature,
                y_feature,
                z_feature,
            ]
        ):
            continue

        z_bounds = transform_lme_bounds(
            bounds[z_feature],
            sample,
        )

        if z_bounds is None:
            continue

        parsed_node_bounds.append((
            bounds[x_feature],
            bounds[y_feature],
            z_bounds,
            weight,
        ))

    if not parsed_node_bounds:
        print(
            f"SKIP cube: no node bounds parsed for "
            f"{sample}, rank={metadata['Rank']}"
        )
        return None

    (
        raw_weight_field,
        smooth_weight_field,
        coverage_field,
    ) = build_cumulative_weight_field(
        node_bounds=parsed_node_bounds,
        grid_size=CUBE_GRID_SIZE,
        smoothing_sigma=CUBE_SMOOTH_SIGMA,
    )

    render_cumulative_weight_field(
        ax=ax,
        smooth_weight_field=smooth_weight_field,
        coverage_field=coverage_field,
        max_abs_single_node_weight=max_abs_single_node_weight,
        cmap=cmap,
        norm=norm,
        grid_size=CUBE_GRID_SIZE,
    )

    if sample == "Full":
        add_full_sample_shell_plane(ax)

    scalar_map = ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    scalar_map.set_array(weights)

    return scalar_map

def plot_one_weight_cube(
    weight_distribution: pd.DataFrame,
    sample: str,
    rank: int,
    section_map: pd.DataFrame,
) -> None:
    """Create one standalone cumulative SDF-weight cube."""
    result = get_cube_data(
        weight_distribution,
        sample,
        rank,
    )

    if result is None:
        return

    sdf, metadata = result

    fig = plt.figure(
        figsize=(10.8, 9.0),
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    scalar_map = draw_weight_cube(
        ax=ax,
        sdf=sdf,
        metadata=metadata,
        sample=sample,
        section_map=section_map,
    )

    if scalar_map is None:
        plt.close(fig)
        return

    colorbar = fig.colorbar(
        scalar_map,
        ax=ax,
        shrink=0.67,
        pad=0.08,
        aspect=24,
    )

    colorbar.set_label(
        "Individual selected-node SDF weight",
        rotation=270,
        labelpad=16,
    )

    note = (
        "Every selected node contributes to the displayed field. "
        "At each location, overlapping node weights are summed before "
        "Gaussian smoothing and rendering."
    )

    if sample == "Full":
        note += (
            " The grey plane marks LME = 30%; the lower region denotes "
            "potential shell contamination."
        )
    else:
        note += (
            " In the cleaned sample, original LME 30%-100% is stretched "
            "to the complete z-axis."
        )

    fig.text(
        0.5,
        0.022,
        note,
        ha="center",
        fontsize=7.7,
    )

    fig.tight_layout(
        rect=(0, 0.05, 1, 1),
    )

    prefix = "09" if sample == "Full" else "10"

    save(
        fig,
        MAIN_DIR,
        f"{prefix}_{sample}_TopSDF_WeightCube_Rank{rank:02d}",
    )

def plot_top_sdf_weight_cubes(
    weight_distribution: pd.DataFrame | None,
    section_map: pd.DataFrame,
) -> None:
    """
    Output standalone cumulative-field cubes.

    Full:
        Top 3.

    Cleaned:
        Top 6.
    """
    if weight_distribution is None or weight_distribution.empty:
        return

    for rank in [1, 2, 3]:
        plot_one_weight_cube(
            weight_distribution=weight_distribution,
            sample="Full",
            rank=rank,
            section_map=section_map,
        )

    for rank in [1, 2, 3, 4, 5, 6]:
        plot_one_weight_cube(
            weight_distribution=weight_distribution,
            sample="Cleaned",
            rank=rank,
            section_map=section_map,
        )

def plot_weight_cubes_combined(
    weight_distribution: pd.DataFrame | None,
    sample: str,
    ranks: list[int],
    section_map: pd.DataFrame,
) -> None:
    """
    Create a combined set of cubes.

    Every subplot has its own local maximum-absolute-weight normalization,
    independent colourbar, and cumulative overlap field.
    """
    if weight_distribution is None or weight_distribution.empty:
        return

    cube_data = []

    for rank in ranks:
        result = get_cube_data(
            weight_distribution,
            sample,
            rank,
        )

        if result is None:
            continue

        sdf, metadata = result
        cube_data.append((rank, sdf, metadata))

    if not cube_data:
        return

    if sample == "Full":
        nrows = 1
        ncols = 3
        figsize = (23.0, 8.2)
        stem = "13_Full_TopSDF_WeightCubes_Combined"
        figure_title = (
            "Full Sample: Top Three Globally Ranked Cumulative SDF Weight Fields"
        )
    else:
        nrows = 2
        ncols = 3
        figsize = (23.0, 15.5)
        stem = "14_Cleaned_TopSDF_WeightCubes_Combined"
        figure_title = (
            "Cleaned Sample: Top Six Globally Ranked Cumulative SDF Weight Fields"
        )

    fig = plt.figure(figsize=figsize)

    for position, (
        rank,
        sdf,
        metadata,
    ) in enumerate(cube_data, start=1):
        ax = fig.add_subplot(
            nrows,
            ncols,
            position,
            projection="3d",
        )

        section_label = section_display_name(
            str(metadata["Section"]),
            section_map,
        )

        description = cube_strategy_description(
            strategy=str(metadata["Strategy"]),
            model_name=str(metadata["Model_Name"]),
            depth_strategy=str(metadata["Depth_Strategy"]),
        )

        diagnostics = cube_weight_diagnostics(
            sdf=sdf,
            metadata=metadata,
        )

        title = (
            f"Rank {rank}: {section_label}\n"
            f"{description}\n"
            f"{cube_diagnostic_text(diagnostics)}"
        )

        scalar_map = draw_weight_cube(
            ax=ax,
            sdf=sdf,
            metadata=metadata,
            sample=sample,
            section_map=section_map,
            title=title,
        )

        if scalar_map is not None:
            colorbar = fig.colorbar(
                scalar_map,
                ax=ax,
                shrink=0.57,
                pad=0.055,
                aspect=18,
            )

            colorbar.ax.tick_params(
                labelsize=6
            )

            colorbar.set_label(
                "Node weight",
                rotation=270,
                labelpad=8,
                fontsize=7,
            )

    for position in range(
        len(cube_data) + 1,
        nrows * ncols + 1,
    ):
        empty_ax = fig.add_subplot(
            nrows,
            ncols,
            position,
        )
        empty_ax.set_axis_off()

    note = (
        "Every selected node is retained. At each grid location, all overlapping "
        "selected-node SDF weights are summed and then lightly smoothed. "
        "Eff. N is the effective number of nodes based on normalized absolute "
        "weights; Sparsity = Eff. N / N."
    )

    if sample == "Full":
        note += (
            " Grey planes indicate LME = 30%; lower regions are the "
            "shell-contamination zone."
        )
    else:
        note += (
            " Original cleaned-sample LME 30%-100% is stretched to the "
            "complete cube height."
        )

    fig.text(
        0.5,
        0.015,
        note,
        ha="center",
        fontsize=7.8,
    )

    fig.suptitle(
        figure_title,
        fontsize=14,
        y=0.985,
    )

    fig.tight_layout(
        rect=(0, 0.04, 1, 0.95),
    )

    save(
        fig,
        MAIN_DIR,
        stem,
    )

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("Step 5: Building final AP-Tree paper figures")
    print("=" * 80)

    MAIN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    APPENDIX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    regression = load_regression()
    xs_r2 = load_xs_r2()
    turnover = load_turnover()
    weight_distribution = load_weight_distribution()

    regression = merge_xs_r2(
        regression=regression,
        xs_r2=xs_r2,
    )

    sections: list[str] = []

    if regression is not None:
        sections.extend(
            regression["Section"]
            .dropna()
            .astype(str)
            .tolist()
        )

    if turnover is not None and "Section" in turnover.columns:
        sections.extend(
            turnover["Section"]
            .dropna()
            .astype(str)
            .tolist()
        )

    if weight_distribution is not None:
        sections.extend(
            weight_distribution["Section"]
            .dropna()
            .astype(str)
            .tolist()
        )

    section_map = build_section_map(sections)

    section_map_path = FIGURE_DIR / "Section_Name_Map.csv"

    section_map.to_csv(
        section_map_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved section mapping: {section_map_path}")

    cleaned_summary = build_summary(
        regression=regression,
        sample="Cleaned",
    )

    plot_depth_weight_performance(
        cleaned_summary=cleaned_summary,
        section_map=section_map,
    )

    plot_overall_boxplots(
        cleaned_summary=cleaned_summary,
    )

    plot_turnover_boxplot(
        turnover=turnover,
    )

    plot_sharpe_by_section(
        regression=regression,
        sample="Full",
        section_map=section_map,
    )

    plot_sharpe_by_section(
        regression=regression,
        sample="Cleaned",
        section_map=section_map,
    )

    for factor_model in MAIN_FACTOR_MODELS:
        plot_factor_evidence_by_section(
            regression=regression,
            factor_model=factor_model,
            section_map=section_map,
            output_folder=MAIN_DIR,
            output_stem=(
                f"04_{factor_model}_Factor_Evidence_By_Section"
            ),
        )

    appendix_stems = {
        "CH3": "A02_CH3_Factor_Evidence_By_Section",
        "FF6": "A03_FF6_Factor_Evidence_By_Section",
        "Carhart4": "A04_Carhart4_Factor_Evidence_By_Section",
    }

    for factor_model in APPENDIX_FACTOR_MODELS:
        plot_factor_evidence_by_section(
            regression=regression,
            factor_model=factor_model,
            section_map=section_map,
            output_folder=APPENDIX_DIR,
            output_stem=appendix_stems[factor_model],
        )

    plot_turnover_by_section(
        turnover=turnover,
        section_map=section_map,
    )

    plot_top_sdf_weight_cubes(
        weight_distribution=weight_distribution,
        section_map=section_map,
    )

    plot_main_factor_evidence_combined(
        regression=regression,
        section_map=section_map,
    )

    plot_full_cleaned_sharpe_combined(
        regression=regression,
        section_map=section_map,
    )

    plot_weight_cubes_combined(
        weight_distribution=weight_distribution,
        sample="Full",
        ranks=[1, 2, 3],
        section_map=section_map,
    )

    plot_weight_cubes_combined(
        weight_distribution=weight_distribution,
        sample="Cleaned",
        ranks=[1, 2, 3, 4, 5, 6],
        section_map=section_map,
    )

    print("=" * 80)
    print(f"Main figures: {MAIN_DIR}")
    print(f"Appendix figures: {APPENDIX_DIR}")
    print(f"Section map: {section_map_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
