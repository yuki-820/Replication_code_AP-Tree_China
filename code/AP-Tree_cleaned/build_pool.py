"""
AP-Tree (cleaned sample) - Build candidate asset pools (tree nodes)
Read data/year/*.csv, filter out bottom 30% market cap each month,
generate AP-Tree nodes for 36 cross-sections (depth=4, binary splits)
Output: output/candidate_pools/AP-Tree_cleaned/Sec*.csv
Data range: January 2011 to December 2025
"""

import pandas as pd
import numpy as np
from itertools import combinations, product
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
    'TREE_DEPTH': 4,
    'BASE_FEATURE': 'mkt_cap',
    'OTHER_FEATURES': ['turnover', 'st_rev', 'r12_2', 'lt_rev', 'idio_vol',
                       'fin_investment', 'fin_op', 'fin_ac', 'beme'],
    'RF_COL': 'rf_rate',
    'Q_NUM': 2,
    'RET_COL': 'ret',
    'WEIGHT_COL': 'mkt_cap',
    'DATE_COL': 'date',
    'OUTPUT_DIR': str(PROJECT_ROOT / 'output' / 'candidate_pools' / 'AP-Tree_cleaned'),
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

def split_node_np(df_arr, feature_idx, q_num):
    feat_vals = df_arr[:, feature_idx]
    groups = ntile_array(feat_vals, q_num)
    children = {}
    for g in range(1, q_num + 1):
        mask = groups == g
        if np.sum(mask) > 0:
            children[g] = mask
    return children

def build_tree_for_month_correct(df_arr, feat_list, depth, q_num, section_name):
    all_node_results = {}
    all_sequences = list(product(feat_list, repeat=depth))
    for seq in all_sequences:
        seq_prefix = '_'.join(seq)
        node_prefix = f"{section_name}-{seq_prefix}"
        n_samples = df_arr.shape[0]
        root_mask = np.ones(n_samples, dtype=bool)
        stack = [(root_mask, 0, "")]
        while stack:
            mask, cur_depth, path_str = stack.pop()
            node_name = f"{node_prefix}_{path_str}" if path_str else node_prefix
            sub_arr = df_arr[mask]
            if sub_arr.shape[0] < 2:
                continue
            weights = np.exp(sub_arr[:, 0])
            rets = sub_arr[:, 1]
            weighted_ret = np.sum(weights * rets) / np.sum(weights) if np.sum(weights) > 0 else np.nan
            all_node_results[node_name] = weighted_ret
            if cur_depth < depth:
                feat_name = seq[cur_depth]
                feature_idx = 2 + feat_list.index(feat_name)
                children = split_node_np(sub_arr, feature_idx, q_num)
                for g, child_mask in children.items():
                    child_global_mask = mask.copy()
                    child_global_mask[mask] = child_mask
                    child_path = f"{path_str}{g}" if path_str else str(g)
                    stack.append((child_global_mask, cur_depth + 1, child_path))
    return all_node_results

def preprocess_monthly_data(config):
    """Preprocess all monthly data: filter out bottom 30% market cap each month"""
    print("Preprocessing monthly data (filter out bottom 30% market cap)...")
    monthly_data = {}
    total_months = 0
    data_folder = Path(config['DATA_FOLDER'])
    for year in range(config['START_YEAR'], config['END_YEAR'] + 1):
        file_path = data_folder / f"{year}.csv"
        if not file_path.exists():
            continue
        df = pd.read_csv(file_path, low_memory=False)
        df.rename(columns=COLUMN_MAP, inplace=True)
        if config['DATE_COL'] not in df.columns:
            continue
        df[config['DATE_COL']] = pd.to_datetime(df[config['DATE_COL']], errors='coerce')
        df['year'] = df[config['DATE_COL']].dt.year
        df['month'] = df[config['DATE_COL']].dt.month
        df = df[df['year'] == year]
        for month in range(1, 13):
            df_month = df[df['month'] == month].copy()
            if df_month.empty:
                continue
            weight_col = config['WEIGHT_COL']
            if weight_col not in df_month.columns:
                continue
            # Filter out bottom 30% by market cap (keep top 70%)
            threshold = df_month[weight_col].quantile(0.3)
            df_filtered = df_month[df_month[weight_col] > threshold].copy()
            date_key = f"{year}-{month:02d}"
            monthly_data[date_key] = df_filtered
            total_months += 1
    print(f"Preprocessing done. {total_months} months filtered and cached.")
    return monthly_data

def generate_ap_tree_for_section(config, feat_list, section_name, monthly_data_cache):
    print(f"  Generating section {section_name} (features: {feat_list})...")
    monthly_tree_data = {}
    for date_key, df_month in monthly_data_cache.items():
        required_cols = [config['WEIGHT_COL'], config['RET_COL']] + feat_list
        if not all(col in df_month.columns for col in required_cols):
            continue
        month_data = df_month[required_cols].dropna()
        if len(month_data) < 2:
            continue
        rf_vals = df_month[config['RF_COL']].dropna()
        rf = rf_vals.iloc[0] if not rf_vals.empty else 0.0
        month_arr = month_data.values
        node_ret_dict = build_tree_for_month_correct(
            month_arr, feat_list, config['TREE_DEPTH'], config['Q_NUM'], section_name
        )
        monthly_tree_data[date_key] = {'node_returns': node_ret_dict, 'rf': rf}
    if not monthly_tree_data:
        print(f"    Section {section_name} has no valid data")
        return None
    # Build DataFrame
    all_nodes = set()
    for data in monthly_tree_data.values():
        all_nodes.update(data['node_returns'].keys())
    all_nodes = sorted(list(all_nodes))
    node_returns_dict = {node: [] for node in all_nodes}
    dates = []
    rf_rates = []
    for date_key in sorted(monthly_tree_data.keys()):
        dates.append(date_key)
        data = monthly_tree_data[date_key]
        rf_rates.append(data['rf'])
        for node in all_nodes:
            ret = data['node_returns'].get(node, np.nan)
            node_returns_dict[node].append(ret)
    df_tree = pd.DataFrame(index=pd.to_datetime(dates))
    for node, returns in node_returns_dict.items():
        df_tree[node] = returns
    df_tree[config['RF_COL']] = rf_rates
    # Remove duplicate and extreme nodes
    node_cols = [col for col in df_tree.columns if col != config['RF_COL']]
    if node_cols:
        df_nodes = df_tree[node_cols]
        dup_mask = df_nodes.T.duplicated(keep='first')
        unique_node_cols = [col for col, dup in zip(node_cols, dup_mask) if not dup]
        depth = config['TREE_DEPTH']
        extreme_mask = []
        for col in unique_node_cols:
            if '-' in col:
                rest = col.split('-', 1)[1]
                parts = rest.split('_')
            else:
                parts = col.split('_')
            feat_parts = [p for p in parts if p in feat_list]
            if len(feat_parts) >= depth and len(set(feat_parts)) == 1:
                extreme_mask.append(True)
            else:
                extreme_mask.append(False)
        final_node_cols = [col for col, is_extreme in zip(unique_node_cols, extreme_mask) if not is_extreme]
        df_tree = df_tree[final_node_cols + [config['RF_COL']]]
    print(f"    {section_name} generated: {len(df_tree)} periods, {df_tree.shape[1]-1} nodes")
    return df_tree

if __name__ == "__main__":
    print("=" * 90)
    print("=== AP-Tree (cleaned sample) - Build candidate asset pools ===")
    print(f"Data range: {CONFIG['START_YEAR']}-01 to {CONFIG['END_YEAR']}-12")
    print("Filter: exclude bottom 30% market cap each month")
    print("=" * 90)
    output_dir = Path(CONFIG['OUTPUT_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)
    # Preprocess data (filter out bottom 30% market cap)
    monthly_data = preprocess_monthly_data(CONFIG)
    if not monthly_data:
        print("Error: No monthly data available after filtering.")
        sys.exit(1)
    depth = CONFIG['TREE_DEPTH']
    combo_size = depth - 2
    other_feats = CONFIG['OTHER_FEATURES']
    combos = list(combinations(other_feats, combo_size))
    print(f"Tree depth: {depth} | Features per section: {1 + combo_size}")
    print(f"Number of cross-sections: {len(combos)}")
    print(f"Output directory: {output_dir}\n")
    start_time = time.time()
    successful = 0
    for idx, combo in enumerate(combos):
        feat_list = [CONFIG['BASE_FEATURE']] + list(combo)
        section_name = f"Sec{idx+1:02d}_{'_'.join(feat_list)}"
        df_tree = generate_ap_tree_for_section(CONFIG, feat_list, section_name, monthly_data)
        if df_tree is not None and df_tree.shape[1] > 1:
            file_path = output_dir / f"{section_name}.csv"
            df_tree.to_csv(file_path, encoding='utf-8-sig')
            print(f"  ✓ {section_name}: {df_tree.shape[1]-1} nodes, {len(df_tree)} periods")
            successful += 1
        else:
            print(f"  ✗ {section_name} failed")
    elapsed = (time.time() - start_time) / 60
    print(f"\nDone! Successful: {successful}/{len(combos)} sections, time: {elapsed:.2f} minutes")