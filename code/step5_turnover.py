"""
Turnover analysis for cleaned sample (bottom 30% market cap filtered monthly).
Computes stock mapping tables and portfolio weights for selected nodes/portfolios
from AP-Tree (cleaned, K=20 & K=40) and TripleSort128 (cleaned) strategies.
Outputs average monthly one-way turnover per section.
"""

import pandas as pd
import numpy as np
from itertools import combinations, product
from pathlib import Path
import warnings
import time

warnings.filterwarnings('ignore')

# ==================== Global configuration ====================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CONFIG = {
    # Data paths
    'DATA_FOLDER': str(PROJECT_ROOT / 'data' / 'year'),
    'AP_MAP_DIR': str(PROJECT_ROOT / 'output' / 'stock_mappings' / 'AP_cleaned'),
    'TRIPLE_MAP_DIR': str(PROJECT_ROOT / 'output' / 'stock_mappings' / 'Triple128_cleaned'),
    'WEIGHTS_OUTPUT_DIR': str(PROJECT_ROOT / 'output' / 'stock_weights_cleaned'),

    # Weight files from cleaned sample pruning (must exist)
    'AP_WEIGHT_FILE_K20': str(PROJECT_ROOT / 'output' / 'pruned' / 'AP-Tree_cleaned' / 'All_Sections_Detailed_Results_kmax20.csv'),
    'AP_WEIGHT_FILE_K40': str(PROJECT_ROOT / 'output' / 'pruned' / 'AP-Tree_cleaned' / 'All_Sections_Detailed_Results_kmax40.csv'),
    'TRIPLE_WEIGHT_FILE': str(PROJECT_ROOT / 'output' / 'pruned' / 'TripleSort128_cleaned' / 'All_TripleSort_Detailed_Results.csv'),

    # Time range (same as used in pruning)
    'START_YEAR': 2011,
    'END_YEAR': 2025,
    'TEST_START': '2021-01-01',
    'TEST_END': '2025-12-31',

    # Model parameters (must match original build scripts)
    'BASE_FEATURE': 'mkt_cap',
    'OTHER_FEATURES': ['turnover', 'st_rev', 'r12_2', 'lt_rev', 'idio_vol',
                       'fin_investment', 'fin_op', 'fin_ac', 'beme'],
    'TREE_DEPTH': 4,
    'Q_NUM': 2,
    'SORT_DIMS': (8, 4, 4),   # for TripleSort128

    # Sections to process (1..36)
    'SECTION_IDS': list(range(1, 37)),

    # Run switches
    'RUN_MAPPING': True,
    'RUN_WEIGHTS': True,
    'RUN_AP20': True,
    'RUN_AP40': True,
    'RUN_TRIPLE': True,
}

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
    '上市公司代码_Comcd': 'stock_id',
    '股票代码_Stkcd': 'stock_id_alt',
}

# ==================== Helper functions ====================
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

def get_stock_id(df):
    if 'stock_id' in df.columns:
        return df['stock_id'].astype(str)
    elif 'stock_id_alt' in df.columns:
        return df['stock_id_alt'].astype(str)
    else:
        raise KeyError("No valid stock id column")

# ==================== AP-Tree mapping generation ====================
def split_node_np(df_arr, feature_idx, q_num):
    feat_vals = df_arr[:, feature_idx]
    groups = ntile_array(feat_vals, q_num)
    children = {}
    for g in range(1, q_num + 1):
        mask = groups == g
        if np.sum(mask) > 0:
            children[g] = mask
    return children

def build_ap_mapping_for_month(df_arr, stock_ids, feat_list, depth, q_num, section_name,
                               date_key, selected_set, mapping_rows):
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
            if node_name in selected_set:
                sub_ids = stock_ids[mask]
                for sid, log_cap in zip(sub_ids, sub_arr[:, 0]):
                    mapping_rows.append((date_key, sid, log_cap, node_name))
            if cur_depth < depth:
                feat_name = seq[cur_depth]
                feature_idx = 2 + feat_list.index(feat_name)
                children = split_node_np(sub_arr, feature_idx, q_num)
                for g, child_mask in children.items():
                    child_global_mask = mask.copy()
                    child_global_mask[mask] = child_mask
                    child_path = f"{path_str}{g}" if path_str else str(g)
                    stack.append((child_global_mask, cur_depth + 1, child_path))

def generate_ap_mappings(config):
    print("\n=== Phase 1: Generate AP-Tree stock mappings (cleaned sample) ===")
    other_feats = config['OTHER_FEATURES']
    combo_size = config['TREE_DEPTH'] - 2
    all_combos = list(combinations(other_feats, combo_size))

    map_dir = Path(config['AP_MAP_DIR'])
    map_dir.mkdir(parents=True, exist_ok=True)

    # Collect selected nodes from both K=20 and K=40 pruning results
    selected_per_section = {}
    for wf_key, wf in [('AP20', config['AP_WEIGHT_FILE_K20']), ('AP40', config['AP_WEIGHT_FILE_K40'])]:
        if not Path(wf).exists():
            print(f"  Warning: weight file not found: {wf}")
            continue
        df_w = pd.read_csv(wf)
        # Determine column names (English or Chinese)
        if 'Type' in df_w.columns:
            type_col = 'Type'
            sec_col = 'Section' if 'Section' in df_w.columns else '截面名称'
            node_col = 'Node_Name' if 'Node_Name' in df_w.columns else '节点名称'
        else:
            type_col = '类型'
            sec_col = '截面名称'
            node_col = '节点名称'
        df_nodes = df_w[df_w[type_col] == 'Selected Node']
        if df_nodes.empty:
            df_nodes = df_w[df_w[type_col] == '选中节点']
        for sec_name, group in df_nodes.groupby(sec_col):
            if sec_name not in selected_per_section:
                selected_per_section[sec_name] = set()
            selected_per_section[sec_name].update(group[node_col].tolist())

    for sec_id in config['SECTION_IDS']:
        combo = all_combos[sec_id - 1]
        feat_list = [config['BASE_FEATURE']] + list(combo)
        section_name = f"Sec{sec_id:02d}_{'_'.join(feat_list)}"
        map_file = map_dir / f"{section_name}_stock_map.csv"
        if map_file.exists():
            print(f"  Skipping {section_name} (map exists)")
            continue
        selected_set = selected_per_section.get(section_name, set())
        if not selected_set:
            print(f"  Warning: No selected nodes for {section_name}, skip")
            continue
        print(f"  Generating mapping for {section_name} (selected nodes: {len(selected_set)})...")
        mapping_rows = []

        data_folder = Path(config['DATA_FOLDER'])
        for year in range(config['START_YEAR'], config['END_YEAR'] + 1):
            file_path = data_folder / f"{year}.csv"
            if not file_path.exists():
                continue
            df = pd.read_csv(file_path, low_memory=False)
            df.rename(columns=COLUMN_MAP, inplace=True)
            if 'date' not in df.columns:
                continue
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df = df[df['year'] == year]
            if 'stock_id' not in df.columns:
                try:
                    df['stock_id'] = get_stock_id(df)
                except KeyError:
                    continue
            for month in range(1, 13):
                df_month = df[df['month'] == month].copy()
                if df_month.empty:
                    continue
                # Filter bottom 30% market cap (cleaned sample)
                if 'mkt_cap' in df_month.columns:
                    threshold = df_month['mkt_cap'].quantile(0.3)
                    df_month = df_month[df_month['mkt_cap'] > threshold].copy()
                else:
                    continue
                date_key = f"{year}-{month:02d}"
                required_cols = ['mkt_cap', 'ret'] + feat_list
                available = [c for c in required_cols if c in df_month.columns]
                if len(available) < len(required_cols):
                    continue
                month_data = df_month[required_cols].dropna()
                if len(month_data) < 2:
                    continue
                stock_ids = df_month.loc[month_data.index, 'stock_id'].values
                month_arr = month_data.values
                build_ap_mapping_for_month(
                    month_arr, stock_ids, feat_list, config['TREE_DEPTH'],
                    config['Q_NUM'], section_name, date_key,
                    selected_set, mapping_rows
                )
        if mapping_rows:
            df_map = pd.DataFrame(mapping_rows, columns=['date', 'stock_id', 'log_mkt_cap', 'node_name'])
            df_map.to_csv(map_file, index=False)
            print(f"    Saved {map_file} ({len(df_map)} records)")
        else:
            print(f"    No data for {section_name}")

# ==================== TripleSort mapping generation ====================
def generate_triple_mappings(config):
    print("\n=== Phase 1: Generate TripleSort128 stock mappings (cleaned sample) ===")
    other_feats = config['OTHER_FEATURES']
    combo_size = 2
    all_combos = list(combinations(other_feats, combo_size))

    map_dir = Path(config['TRIPLE_MAP_DIR'])
    map_dir.mkdir(parents=True, exist_ok=True)

    dims = config['SORT_DIMS']
    wf = config['TRIPLE_WEIGHT_FILE']
    if not Path(wf).exists():
        print(f"Error: Triple weight file not found: {wf}")
        return
    df_weights = pd.read_csv(wf)
    # Detect columns
    if 'Type' in df_weights.columns:
        type_col = 'Type'
        sec_col = 'Section' if 'Section' in df_weights.columns else '截面名称'
        node_col = 'Portfolio_Name' if 'Portfolio_Name' in df_weights.columns else '组合名称'
    else:
        type_col = '类型'
        sec_col = '截面名称'
        node_col = '组合名称'
    df_nodes = df_weights[df_weights[type_col] == 'Selected Node']
    if df_nodes.empty:
        df_nodes = df_weights[df_weights[type_col] == '选中节点']

    for sec_id in config['SECTION_IDS']:
        combo = all_combos[sec_id - 1]
        feat_list = [config['BASE_FEATURE']] + list(combo)
        section_name = f"Sec{sec_id:02d}_{'_'.join(feat_list)}"
        triple_section_name = section_name + "_triplesort"
        map_file = map_dir / f"{section_name}_triplesort_stock_map.csv"
        if map_file.exists():
            print(f"  Skipping {section_name} (map exists)")
            continue
        sec_df = df_nodes[df_nodes[sec_col] == triple_section_name]
        selected_set = set(sec_df[node_col].tolist())
        if not selected_set:
            print(f"  Warning: No selected portfolios for {section_name}, skip")
            continue
        print(f"  Generating mapping for {section_name} (selected: {len(selected_set)})...")
        mapping_rows = []

        data_folder = Path(config['DATA_FOLDER'])
        for year in range(config['START_YEAR'], config['END_YEAR'] + 1):
            file_path = data_folder / f"{year}.csv"
            if not file_path.exists():
                continue
            df = pd.read_csv(file_path, low_memory=False)
            df.rename(columns=COLUMN_MAP, inplace=True)
            if 'date' not in df.columns:
                continue
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df = df[df['year'] == year]
            if 'stock_id' not in df.columns:
                try:
                    df['stock_id'] = get_stock_id(df)
                except KeyError:
                    continue
            for month in range(1, 13):
                df_month = df[df['month'] == month].copy()
                if df_month.empty:
                    continue
                # Filter bottom 30% market cap
                if 'mkt_cap' in df_month.columns:
                    threshold = df_month['mkt_cap'].quantile(0.3)
                    df_month = df_month[df_month['mkt_cap'] > threshold].copy()
                else:
                    continue
                date_key = f"{year}-{month:02d}"
                required_cols = ['mkt_cap', 'ret'] + feat_list
                available = [c for c in required_cols if c in df_month.columns]
                if len(available) < len(required_cols):
                    continue
                month_data = df_month[required_cols].dropna()
                if len(month_data) < 2:
                    continue
                stock_ids = df_month.loc[month_data.index, 'stock_id'].values
                month_arr = month_data.values
                f1 = month_arr[:, 2]
                f2 = month_arr[:, 3]
                f3 = month_arr[:, 4]
                g1 = ntile_array(f1, dims[0])
                g2 = ntile_array(f2, dims[1])
                g3 = ntile_array(f3, dims[2])
                for i in range(1, dims[0]+1):
                    for j in range(1, dims[1]+1):
                        for k in range(1, dims[2]+1):
                            group_id = f"{i}_{j}_{k}"
                            if group_id not in selected_set:
                                continue
                            mask = (g1 == i) & (g2 == j) & (g3 == k)
                            if np.sum(mask) == 0:
                                continue
                            sub_ids = stock_ids[mask]
                            sub_log_cap = month_arr[mask, 0]
                            for sid, log_cap in zip(sub_ids, sub_log_cap):
                                mapping_rows.append((date_key, sid, log_cap, group_id))
        if mapping_rows:
            df_map = pd.DataFrame(mapping_rows, columns=['date', 'stock_id', 'log_mkt_cap', 'group_id'])
            df_map.to_csv(map_file, index=False)
            print(f"    Saved {map_file} ({len(df_map)} records)")
        else:
            print(f"    No data for {section_name}")

# ==================== Weight table & turnover calculation ====================
def read_node_weights_ap(weight_file_path, section_name):
    df = pd.read_csv(weight_file_path)
    if 'Type' in df.columns:
        type_col = 'Type'
        sec_col = 'Section' if 'Section' in df.columns else '截面名称'
        node_col = 'Node_Name' if 'Node_Name' in df.columns else '节点名称'
        weight_col = 'Weight' if 'Weight' in df.columns else '权重'
    else:
        type_col = '类型'
        sec_col = '截面名称'
        node_col = '节点名称'
        weight_col = '权重'
    df_nodes = df[df[type_col] == 'Selected Node']
    if df_nodes.empty:
        df_nodes = df[df[type_col] == '选中节点']
    sec_df = df_nodes[df_nodes[sec_col] == section_name]
    weights = {}
    for _, row in sec_df.iterrows():
        weights[row[node_col]] = row[weight_col]
    return weights

def read_combo_weights_triple(weight_file_path, section_name):
    df = pd.read_csv(weight_file_path)
    if 'Type' in df.columns:
        type_col = 'Type'
        sec_col = 'Section' if 'Section' in df.columns else '截面名称'
        node_col = 'Portfolio_Name' if 'Portfolio_Name' in df.columns else '组合名称'
        weight_col = 'Weight' if 'Weight' in df.columns else '权重'
    else:
        type_col = '类型'
        sec_col = '截面名称'
        node_col = '组合名称'
        weight_col = '权重'
    df_nodes = df[df[type_col] == 'Selected Node']
    if df_nodes.empty:
        df_nodes = df[df[type_col] == '选中节点']
    target_name = section_name + "_triplesort"
    sec_df = df_nodes[df_nodes[sec_col] == target_name]
    weights = {}
    for _, row in sec_df.iterrows():
        weights[row[node_col]] = row[weight_col]
    return weights

def build_weights_table(model_type, section_name, map_dir, weight_dict, test_start, test_end):
    if model_type == 'AP':
        map_file = Path(map_dir) / f"{section_name}_stock_map.csv"
        group_col = 'node_name'
    else:
        map_file = Path(map_dir) / f"{section_name}_triplesort_stock_map.csv"
        group_col = 'group_id'
    if not map_file.exists():
        return None
    df_map = pd.read_csv(map_file)
    df_map['date'] = pd.to_datetime(df_map['date'])
    mask = (df_map['date'] >= test_start) & (df_map['date'] <= test_end)
    df_test = df_map.loc[mask].copy()
    if df_test.empty:
        return None
    df_test['mktcap'] = np.exp(df_test['log_mkt_cap'])
    grouped = df_test.groupby(['date', group_col])['mktcap'].sum().reset_index()
    grouped.rename(columns={'mktcap': 'group_mktcap'}, inplace=True)
    df_test = df_test.merge(grouped, on=['date', group_col], how='left')
    df_test['weight_in_group'] = df_test['mktcap'] / df_test['group_mktcap']
    df_test = df_test[df_test[group_col].isin(weight_dict.keys())].copy()
    if df_test.empty:
        return None
    df_test['config_weight'] = df_test[group_col].map(weight_dict)
    df_test['raw_weight'] = df_test['weight_in_group'] * df_test['config_weight']
    # No additional normalization
    pivot = df_test.pivot_table(index='stock_id', columns='date', values='raw_weight',
                                aggfunc='sum', fill_value=0.0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot = pivot.astype(float)
    return pivot

def compute_turnover(weights_df):
    if weights_df.empty or weights_df.shape[1] < 2:
        return np.nan
    dates = weights_df.columns
    n_months = len(dates) - 1
    total = 0.0
    for i in range(1, len(dates)):
        prev = weights_df[dates[i-1]].fillna(0.0)
        curr = weights_df[dates[i]].fillna(0.0)
        total += (curr - prev).abs().sum() / 2.0
    return total / n_months if n_months > 0 else np.nan

def generate_all_weights(config):
    print("\n=== Phase 2: Build weight tables and compute turnover (cleaned sample) ===")
    out_dir = Path(config['WEIGHTS_OUTPUT_DIR'])
    out_dir.mkdir(parents=True, exist_ok=True)

    test_start = config['TEST_START']
    test_end = config['TEST_END']
    other_feats = config['OTHER_FEATURES']
    combo_size = 2
    all_combos = list(combinations(other_feats, combo_size))

    turnover_records = []

    for sec_id in config['SECTION_IDS']:
        combo = all_combos[sec_id - 1]
        feat_list = [config['BASE_FEATURE']] + list(combo)
        section_name = f"Sec{sec_id:02d}_{'_'.join(feat_list)}"
        print(f"\nProcessing section: {section_name}")

        # AP K=20 cleaned
        if config['RUN_AP20']:
            wf = config['AP_WEIGHT_FILE_K20']
            if Path(wf).exists():
                weights = read_node_weights_ap(wf, section_name)
                if weights:
                    df_w = build_weights_table('AP', section_name, config['AP_MAP_DIR'],
                                                weights, test_start, test_end)
                    if df_w is not None:
                        out_path = out_dir / f"AP20_cleaned_{section_name}_weights.csv"
                        df_w.to_csv(out_path, float_format='%.10f')
                        print(f"  ✓ AP20: {df_w.shape[0]} stocks × {df_w.shape[1]} months")
                        avg_turn = compute_turnover(df_w)
                        turnover_records.append({
                            'Model': 'AP20_cleaned',
                            'Section': section_name,
                            'Average_Monthly_Turnover': avg_turn
                        })
                    else:
                        print(f"  ⚠ AP20: no weight table")
                else:
                    print(f"  ⚠ AP20: no selected nodes")
            else:
                print(f"  ✗ AP20 weight file missing")

        # AP K=40 cleaned
        if config['RUN_AP40']:
            wf = config['AP_WEIGHT_FILE_K40']
            if Path(wf).exists():
                weights = read_node_weights_ap(wf, section_name)
                if weights:
                    df_w = build_weights_table('AP', section_name, config['AP_MAP_DIR'],
                                                weights, test_start, test_end)
                    if df_w is not None:
                        out_path = out_dir / f"AP40_cleaned_{section_name}_weights.csv"
                        df_w.to_csv(out_path, float_format='%.10f')
                        print(f"  ✓ AP40: {df_w.shape[0]} stocks × {df_w.shape[1]} months")
                        avg_turn = compute_turnover(df_w)
                        turnover_records.append({
                            'Model': 'AP40_cleaned',
                            'Section': section_name,
                            'Average_Monthly_Turnover': avg_turn
                        })
                    else:
                        print(f"  ⚠ AP40: no weight table")
                else:
                    print(f"  ⚠ AP40: no selected nodes")
            else:
                print(f"  ✗ AP40 weight file missing")

        # TripleSort128 cleaned
        if config['RUN_TRIPLE']:
            wf = config['TRIPLE_WEIGHT_FILE']
            if Path(wf).exists():
                weights = read_combo_weights_triple(wf, section_name)
                if weights:
                    df_w = build_weights_table('Triple', section_name, config['TRIPLE_MAP_DIR'],
                                                weights, test_start, test_end)
                    if df_w is not None:
                        out_path = out_dir / f"Triple128_cleaned_{section_name}_weights.csv"
                        df_w.to_csv(out_path, float_format='%.10f')
                        print(f"  ✓ Triple128: {df_w.shape[0]} stocks × {df_w.shape[1]} months")
                        avg_turn = compute_turnover(df_w)
                        turnover_records.append({
                            'Model': 'Triple128_cleaned',
                            'Section': section_name,
                            'Average_Monthly_Turnover': avg_turn
                        })
                    else:
                        print(f"  ⚠ Triple128: no weight table")
                else:
                    print(f"  ⚠ Triple128: no selected portfolios")
            else:
                print(f"  ✗ Triple128 weight file missing")

    if turnover_records:
        df_turn = pd.DataFrame(turnover_records)
        summary_path = out_dir / "Average_Turnover_Summary_Cleaned.csv"
        df_turn.to_csv(summary_path, index=False, float_format='%.6f')
        print(f"\nTurnover summary saved to {summary_path}")
    else:
        print("\nNo turnover data generated.")

# ==================== Main ====================
if __name__ == "__main__":
    print("=" * 80)
    print("Turnover Analysis for Cleaned Sample (bottom 30% filtered)")
    print("=" * 80)

    start_time = time.time()

    if CONFIG['RUN_MAPPING']:
        generate_ap_mappings(CONFIG)
        generate_triple_mappings(CONFIG)

    if CONFIG['RUN_WEIGHTS']:
        generate_all_weights(CONFIG)

    elapsed = (time.time() - start_time) / 60
    print(f"\nAll tasks completed in {elapsed:.2f} minutes.")