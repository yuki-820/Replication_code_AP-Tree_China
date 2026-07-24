"""
Plotting script: monthly Sharpe, alpha, and R² across sections for various strategies.
Each plot has its own section ordering, based on the corresponding metric of the
reference line (AP20 dw_power2.0 for full/cleaned; AP_long5 dw_power2.0 for long‑only).
Reads precomputed regression results from All_Regression_Results.csv.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

REGRESSION_FILE = PROJECT_ROOT / 'output' / 'tables' / 'All_Regression_Results.csv'
OUTPUT_FIG_DIR = PROJECT_ROOT / 'output' / 'figures'

FACTOR_MODELS = ['CH3', 'CH4', 'FF5', 'FF6', 'Carhart4']

SAMPLE_STRATEGIES = {
    'full_sample': {
        'TripleSort64': 'Triple64',
        'APTree_K20': 'AP20',
        'APTree_K40': 'AP40',
    },
    'cleaned_sample': {
        'TripleSort128_clean': 'Triple128',
        'APTree_clean_K20': 'AP20',
        'APTree_clean_K40': 'AP40',
    },
    'longonly_sample': {
        'TripleSort128_longonly': 'Triple128_long',
        'APTree_longonly_K5': 'AP_long5',
    }
}

# Reference line for ordering per sample type
REFERENCE = {
    'full_sample': ('APTree_K20', 'dw_power2.0'),
    'cleaned_sample': ('APTree_clean_K20', 'dw_power2.0'),
    'longonly_sample': ('APTree_longonly_K5', 'dw_power2.0'),
}

FEATURE_MAP = {
    'mkt_cap': 'LME', 'turnover': 'Lturnover', 'st_rev': 'ST_Rev',
    'r12_2': 'r12_2', 'lt_rev': 'LT_Rev', 'idio_vol': 'IdioVol',
    'fin_investment': 'Fin_Investment', 'fin_op': 'Fin_OP',
    'fin_ac': 'Fin_AC', 'beme': 'BEME'
}

def make_short_labels(section_names):
    labels = []
    for s in section_names:
        parts = s.split('_')
        if parts[0].startswith('Sec'):
            feat_parts = parts[1:]
        else:
            feat_parts = parts
        labels.append(', '.join([FEATURE_MAP.get(f, f) for f in feat_parts]))
    return labels

def extract_base_section(portfolio_name):
    name = re.sub(r'_dw_power\d+\.?\d*$', '', portfolio_name)
    name = re.sub(r'_triplesort$', '', name)
    return name

def plot_metric(series_dict, section_labels, ylabel, title, filepath):
    fig, ax = plt.subplots(figsize=(20, 6))
    for label, values in series_dict.items():
        ax.plot(range(len(section_labels)), values, marker='o', markersize=3, linewidth=1.2, label=label)
    ax.set_xticks(range(len(section_labels)))
    ax.set_xticklabels(section_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

def build_line_data(sample_df, strat_map):
    """
    Returns a dict: line_label -> DataFrame (index=base_section, columns=metrics)
    Metrics: sharpe, alpha_CH3, R2_CH3, ...
    """
    # We need to pivot the long-format df into wide format per portfolio
    # First, create a unique portfolio identifier (base_section + weight_scheme?)
    # Actually each combination of (Strategy, Weight_Scheme, base_section) is a line.
    # We'll build a wide DataFrame: index = (Strategy, Weight_Scheme, base_section)
    # Columns = Test_Monthly_Sharpe, Alpha_CH3, R2_CH3, ...
    # Then group by (Strategy, Weight_Scheme) to create separate lines.

    sample_df = sample_df.copy()
    sample_df['base'] = sample_df['Portfolio'].apply(extract_base_section)

    # Pivot: for each (Strategy, Weight_Scheme, base), get the metrics
    # We can create multi-index columns by combining metric and model
    pivot = sample_df.pivot_table(index=['Strategy', 'Weight_Scheme', 'base'],
                                  columns='Model',
                                  values=['Test_Monthly_Sharpe', 'Alpha_annual', 'R2'],
                                  aggfunc='first')
    # Flatten columns: e.g., ('Test_Monthly_Sharpe', 'CH4') -> 'Sharpe' (since sharpe is same across models, take first)
    # Alpha and R2 become 'alpha_CH3', 'R2_CH3', etc.
    # We'll take Sharpe from the first model
    pivot.columns = [f'{col[0]}_{col[1]}' for col in pivot.columns.values]
    # Actually Sharpe is repeated, we can take any
    sharpe_col = [c for c in pivot.columns if c.startswith('Test_Monthly_Sharpe_')][0]
    pivot['sharpe'] = pivot[sharpe_col]
    # Drop all original sharpe cols
    pivot = pivot.drop(columns=[c for c in pivot.columns if c.startswith('Test_Monthly_Sharpe_')])
    # Rename Alpha_annual_XXX -> alpha_XXX, R2_XXX -> R2_XXX
    rename_dict = {}
    for c in pivot.columns:
        if c.startswith('Alpha_annual_'):
            model = c.split('_', 2)[2]
            rename_dict[c] = f'alpha_{model}'
        elif c.startswith('R2_'):
            model = c.split('_', 1)[1]
            rename_dict[c] = f'R2_{model}'
    pivot.rename(columns=rename_dict, inplace=True)

    # Now group by (Strategy, Weight_Scheme) to create lines
    lines = {}
    for (strat, wscheme), group in pivot.groupby(level=[0, 1]):
        # Determine display label
        label = strat_map[strat]
        if wscheme not in ['', 'none']:
            label = f"{label} {wscheme}"
        # We only need the base section as index
        df_line = group.droplevel([0, 1])
        lines[label] = df_line
    return lines

def main():
    if not REGRESSION_FILE.exists():
        print(f"Error: {REGRESSION_FILE} not found.")
        sys.exit(1)

    print("Loading regression results...")
    df_all = pd.read_csv(REGRESSION_FILE, encoding='utf-8-sig')
    required = ['Strategy', 'Weight_Scheme', 'Portfolio', 'Feature',
                'Test_Monthly_Sharpe', 'Model', 'Alpha_annual', 'R2']
    for col in required:
        if col not in df_all.columns:
            print(f"Error: Missing column {col}")
            sys.exit(1)

    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    for sample_type, strat_map in SAMPLE_STRATEGIES.items():
        print(f"\n{'='*60}")
        print(f"Processing {sample_type}")
        sample_dir = OUTPUT_FIG_DIR / sample_type
        sample_dir.mkdir(parents=True, exist_ok=True)

        sample_strats = list(strat_map.keys())
        sample_df = df_all[df_all['Strategy'].isin(sample_strats)]
        if sample_df.empty:
            print("  No data, skipping.")
            continue

        # Build all lines
        lines = build_line_data(sample_df, strat_map)
        if not lines:
            print("  No lines generated.")
            continue

        # Reference line for ordering
        ref_strat, ref_scheme = REFERENCE[sample_type]
        ref_label = strat_map[ref_strat]
        if ref_scheme not in ['', 'none']:
            ref_label = f"{ref_label} {ref_scheme}"
        if ref_label not in lines:
            print(f"  Warning: Reference line '{ref_label}' not found. Skipping plots for {sample_type}.")
            continue
        ref_df = lines[ref_label]

        # ---- Plot Sharpe ----
        # Order by reference Sharpe
        ref_sharpe = ref_df['sharpe'].dropna()
        order_secs = ref_sharpe.sort_values().index.tolist()
        labels_x = make_short_labels(order_secs)
        plot_dict = {}
        for line_label, df_line in lines.items():
            plot_dict[line_label] = df_line['sharpe'].reindex(order_secs).tolist()
        plot_metric(plot_dict, labels_x, 'Monthly Sharpe',
                    f'Monthly Sharpe - {sample_type}',
                    sample_dir / f'{sample_type}_monthly_sharpe.png')

        # ---- Plot Alpha for each model ----
        for model in FACTOR_MODELS:
            alpha_col = f'alpha_{model}'
            if alpha_col not in ref_df.columns:
                continue
            ref_alpha = ref_df[alpha_col].dropna()
            order_secs = ref_alpha.sort_values().index.tolist()
            labels_x = make_short_labels(order_secs)
            plot_dict = {}
            for line_label, df_line in lines.items():
                if alpha_col in df_line.columns:
                    plot_dict[line_label] = df_line[alpha_col].reindex(order_secs).tolist()
                else:
                    plot_dict[line_label] = [np.nan] * len(order_secs)
            plot_metric(plot_dict, labels_x, f'{model} Annual Alpha',
                        f'{model} Alpha - {sample_type}',
                        sample_dir / f'{sample_type}_alpha_{model}.png')

        # ---- Plot R² for each model ----
        for model in FACTOR_MODELS:
            r2_col = f'R2_{model}'
            if r2_col not in ref_df.columns:
                continue
            ref_r2 = ref_df[r2_col].dropna()
            order_secs = ref_r2.sort_values().index.tolist()
            labels_x = make_short_labels(order_secs)
            plot_dict = {}
            for line_label, df_line in lines.items():
                if r2_col in df_line.columns:
                    plot_dict[line_label] = df_line[r2_col].reindex(order_secs).tolist()
                else:
                    plot_dict[line_label] = [np.nan] * len(order_secs)
            plot_metric(plot_dict, labels_x, f'{model} R²',
                        f'{model} R² - {sample_type}',
                        sample_dir / f'{sample_type}_R2_{model}.png')

    print("\nAll plots generated successfully.")

if __name__ == "__main__":
    main()
