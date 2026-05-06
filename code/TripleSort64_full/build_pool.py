"""
TripleSort64 (full sample) - Build candidate asset pools (4x4x4 triple-sorted portfolios)
Read data/year/*.csv, generate 36 cross-sections of 64 portfolios each
Output: output/candidate_pools/TripleSort64_full/Sec*_triplesort.csv
Data range: January 2011 to December 2025
"""

import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
import warnings
import time
import sys

warnings.filterwarnings('ignore')

# ==================== Dynamic path config ====================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Column name mapping (Chinese -> English)
COLUMN_MAP = {
    '日期_Date': 'date',
    '月无风险收益率_Monrfret': 'rf_rate',
    '月收益率_Monret': 'ret',
    'LME': 'mkt_cap',
    'Lturnover': 'turnover',
    'ST_Rev': 'st_rev',
    'r12_2': 'r12_2',
    'LT_Rev': 'lt_rev',
    'IdioVol': 'idio_vol',
    'Fin_Investment': 'fin_investment',
    'Fin_OP': 'fin_op',
    'Fin_AC': 'fin_ac',
    'BEME': 'beme',
}

CONFIG = {
    'DATA_FOLDER': str(PROJECT_ROOT / 'data' / 'year'),
    'START_YEAR': 2011,
    'END_YEAR': 2025,
    'BASE_FEATURE': 'mkt_cap',                # English name
    'OTHER_FEATURES': ['turnover', 'st_rev', 'r12_2', 'lt_rev', 'idio_vol',
                       'fin_investment', 'fin_op', 'fin_ac', 'beme'],
    'RF_COL': 'rf_rate',
    'RET_COL': 'ret',
    'WEIGHT_COL': 'mkt_cap',
    'DATE_COL': 'date',
    'OUTPUT_DIR': str(PROJECT_ROOT / 'output' / 'candidate_pools' / 'TripleSort64_full'),
    'SORT_DIMS': (4, 4, 4),
}

def ntile_array(arr, q):
    out = np.full(len(arr), np.nan, dtype=float)
    nonnull_idx = ~np.isnan(arr)
    nonnull_vals = arr[nonnull_idx]
    n = len(nonnull_vals)
    if n == 0:
        return out
    if np.unique(nonnull_vals).size == 1:
        out[nonnull_idx] = 1
    else:
        order = np.argsort(nonnull_vals)
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n) + 1
        bins = np.ceil(ranks / (n / q)).astype(int)
        bins[bins > q] = q
        out[nonnull_idx] = bins
    return out.astype(int)

def triple_sort_portfolio(month_arr, feat_indices, weight_idx, ret_idx, dims):
    weights = np.exp(month_arr[:, weight_idx])
    rets = month_arr[:, ret_idx]
    f1_vals = month_arr[:, feat_indices[0]]
    f2_vals = month_arr[:, feat_indices[1]]
    f3_vals = month_arr[:, feat_indices[2]]

    g1 = ntile_array(f1_vals, dims[0])
    g2 = ntile_array(f2_vals, dims[1])
    g3 = ntile_array(f3_vals, dims[2])

    port_rets = {}
    for i in range(1, dims[0]+1):
        for j in range(1, dims[1]+1):
            for k in range(1, dims[2]+1):
                mask = (g1 == i) & (g2 == j) & (g3 == k)
                if np.sum(mask) == 0:
                    port_rets[(i,j,k)] = np.nan
                else:
                    w = weights[mask]
                    r = rets[mask]
                    port_ret = np.sum(w * r) / np.sum(w) if np.sum(w) > 0 else np.nan
                    port_rets[(i,j,k)] = port_ret
    return port_rets

def generate_triplesort_for_section(config, feat_list, section_name):
    print(f"  Generating section {section_name} ...")
    monthly_portfolios = {}
    data_folder = Path(config['DATA_FOLDER'])

    for year in range(config['START_YEAR'], config['END_YEAR'] + 1):
        file_path = data_folder / f"{year}.csv"
        if not file_path.exists():
            if year == config['START_YEAR']:
                print(f"    Warning: data file not found {file_path}")
            continue

        df = pd.read_csv(file_path, low_memory=False)
        # Rename columns to English
        df.rename(columns=COLUMN_MAP, inplace=True)

        if config['DATE_COL'] not in df.columns:
            print(f"    Error: missing column '{config['DATE_COL']}' in {file_path}")
            return None
        df[config['DATE_COL']] = pd.to_datetime(df[config['DATE_COL']], errors='coerce')
        df['year'] = df[config['DATE_COL']].dt.year
        df['month'] = df[config['DATE_COL']].dt.month
        df = df[df['year'] == year]

        for month in range(1, 13):
            df_month = df[df['month'] == month].copy()
            if df_month.empty:
                continue

            date_key = f"{year}-{month:02d}"
            rf_vals = df_month[config['RF_COL']].dropna()
            rf = rf_vals.iloc[0] if not rf_vals.empty else 0.0

            required_cols = [config['WEIGHT_COL'], config['RET_COL']] + feat_list
            available_cols = [col for col in required_cols if col in df_month.columns]
            if len(available_cols) < len(required_cols):
                missing = set(required_cols) - set(available_cols)
                if year == config['START_YEAR'] and month == 1:
                    print(f"    Missing columns: {missing}, skipping section")
                continue

            month_data = df_month[required_cols].dropna()
            if len(month_data) < 2:
                continue

            month_arr = month_data.values
            weight_idx = 0
            ret_idx = 1
            feat_indices = [2, 3, 4]
            port_rets_dict = triple_sort_portfolio(month_arr, feat_indices, weight_idx, ret_idx, config['SORT_DIMS'])

            monthly_dict = {}
            for (i,j,k), ret_val in port_rets_dict.items():
                col_name = f"{i}_{j}_{k}"
                monthly_dict[col_name] = (ret_val - rf) if not np.isnan(ret_val) else np.nan
            monthly_portfolios[date_key] = monthly_dict

    if not monthly_portfolios:
        print(f"  Section {section_name} has no valid data")
        return None

    dates = sorted(monthly_portfolios.keys())
    all_cols = sorted({col for d in monthly_portfolios.values() for col in d.keys()})
    data = []
    for date in dates:
        row = [date]
        row.extend([monthly_portfolios[date].get(col, np.nan) for col in all_cols])
        data.append(row)

    df_out = pd.DataFrame(data, columns=['Date'] + all_cols)
    df_out['Date'] = pd.to_datetime(df_out['Date'])
    df_out = df_out.set_index('Date')
    return df_out

if __name__ == "__main__":
    print("=" * 90)
    print("=== TripleSort64 (full sample) - Build candidate asset pools ===")
    print(f"Script location: {SCRIPT_DIR}")
    print(f"Data directory: {CONFIG['DATA_FOLDER']}")
    print(f"Output directory: {CONFIG['OUTPUT_DIR']}")
    print("=" * 90)

    data_path = Path(CONFIG['DATA_FOLDER'])
    if not data_path.exists():
        print(f"Error: data directory does not exist -> {data_path}")
        sys.exit(1)

    existing_years = [f.name for f in data_path.glob("*.csv")]
    print(f"Found year files: {sorted(existing_years)[:5]}... (total {len(existing_years)})")

    output_dir = Path(CONFIG['OUTPUT_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)

    depth = 4
    combo_size = depth - 2
    other_feats = CONFIG['OTHER_FEATURES']
    combos = list(combinations(other_feats, combo_size))

    print(f"Number of cross-sections: {len(combos)}")
    print(f"Sorting dimensions: {CONFIG['SORT_DIMS'][0]}x{CONFIG['SORT_DIMS'][1]}x{CONFIG['SORT_DIMS'][2]} = {CONFIG['SORT_DIMS'][0]*CONFIG['SORT_DIMS'][1]*CONFIG['SORT_DIMS'][2]} portfolios\n")

    start = time.time()
    success = 0
    for idx, combo in enumerate(combos):
        feat_list = [CONFIG['BASE_FEATURE']] + list(combo)
        section_name = f"Sec{idx+1:02d}_{'_'.join(feat_list)}"
        df_triple = generate_triplesort_for_section(CONFIG, feat_list, section_name)
        if df_triple is not None and df_triple.shape[1] > 0:
            out_file = output_dir / f"{section_name}_triplesort.csv"
            df_triple.to_csv(out_file)
            print(f"  ✓ {section_name}: {df_triple.shape[1]} portfolios, {len(df_triple)} periods")
            success += 1
        else:
            print(f"  ✗ {section_name} failed")

    elapsed = (time.time() - start) / 60
    print(f"\nDone! Successful: {success}/{len(combos)} sections, time: {elapsed:.2f} minutes")