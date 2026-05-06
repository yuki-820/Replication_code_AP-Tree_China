"""
Comprehensive factor model testing for all portfolio strategies.

Reads test excess returns from pruned portfolios, performs time-series regressions
for CH3, CH4, FF5, FF6, Carhart4 models, and exports required tables.
All results are saved to output/tables/ in CSV format.
Handles missing data gracefully.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from scipy.stats import f as f_dist
import warnings

warnings.filterwarnings('ignore')

# ==================== Dynamic path configuration ====================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CONFIG = {
    'factor_dir': str(PROJECT_ROOT / 'data' / 'factors'),
    'output_dir': str(PROJECT_ROOT / 'output' / 'tables'),
    'models': {
        'CH3': {
            'file': 'CH3_factors_monthly_202602.xlsx',
            'sheet': 'Returnseries',
            'mapping': {'mktrf': 'Mkt-RF', 'SMB': 'SMB', 'VMG': 'VMG'}
        },
        'CH4': {
            'file': 'CH4_factors_monthly_202602.xlsx',
            'sheet': 'Returnseries',
            'mapping': {'mktrf': 'Mkt-RF', 'SMB': 'SMB', 'VMG': 'VMG', 'PMO': 'PMO'}
        },
        'FF5': {
            'file': 'fivefactor_monthly.csv',
            'sheet': None,
            'mapping': {'mkt_rf': 'Mkt-RF', 'smb': 'SMB', 'hml': 'HML', 'rmw': 'RMW', 'cma': 'CMA'}
        },
        'FF6': {
            'file': 'fivefactor_monthly.csv',
            'sheet': None,
            'mapping': {'mkt_rf': 'Mkt-RF', 'smb': 'SMB', 'hml': 'HML', 'rmw': 'RMW', 'cma': 'CMA', 'umd': 'UMD'}
        },
        'Carhart4': {
            'file': 'fivefactor_monthly.csv',
            'sheet': None,
            'mapping': {'mkt_rf': 'Mkt-RF', 'smb': 'SMB', 'hml': 'HML', 'umd': 'UMD'}
        }
    },
    'strategies': {
        'TripleSort64': 'output/pruned/TripleSort64_full/Test_Excess_Returns_TripleSort64.csv',
        'APTree_K20': 'output/pruned/AP-Tree_full/Test_Excess_Returns_kmax20.csv',
        'APTree_K40': 'output/pruned/AP-Tree_full/Test_Excess_Returns_kmax40.csv',
        'TripleSort128_clean': 'output/pruned/TripleSort128_cleaned/Test_Excess_Returns_TripleSort128.csv',
        'APTree_clean_K20': 'output/pruned/AP-Tree_cleaned/Test_Excess_Returns_kmax20.csv',
        'APTree_clean_K40': 'output/pruned/AP-Tree_cleaned/Test_Excess_Returns_kmax40.csv',
        'TripleSort128_longonly': 'output/pruned/TripleSort128_cleaned_longonly/Test_Excess_Returns_TripleSort128_longonly.csv',
        'APTree_longonly_K5': 'output/pruned/AP-Tree_cleaned_longonly/Test_Excess_Returns_APT_longonly.csv',
    },
    'nw_lags': None,
}

# Feature name mapping: internal -> display
FEATURE_MAP = {
    'mkt_cap': 'LME',
    'turnover': 'Lturnover',
    'st_rev': 'ST_Rev',
    'r12_2': 'r12_2',
    'lt_rev': 'LT_Rev',
    'idio_vol': 'IdioVol',
    'fin_investment': 'Fin_Investment',
    'fin_op': 'Fin_OP',
    'fin_ac': 'Fin_AC',
    'beme': 'BEME',
}

def load_factors(model_name):
    cfg = CONFIG['models'][model_name]
    file_path = Path(CONFIG['factor_dir']) / cfg['file']
    if cfg['file'].endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path, sheet_name=cfg['sheet'])
    date_col = None
    for col in ['日期_Date', 'mnthdt', 'trdmn']:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        raise ValueError(f"Date column not found in {cfg['file']}")
    if model_name in ['CH3', 'CH4']:
        df[date_col] = pd.to_datetime(df[date_col].astype(str).str.replace('.0', ''), format='%Y%m%d', errors='coerce')
    elif date_col == 'trdmn':
        df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m', errors='coerce')
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['year_month'] = df[date_col].dt.strftime('%Y-%m')
    df = df.set_index('year_month')
    available = [c for c in cfg['mapping'].keys() if c in df.columns]
    df = df[available].rename(columns=cfg['mapping'])
    df = df.astype(float).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df

def run_factor_regression(strategy_ret, factors, nw_lags=None):
    merged = pd.concat([strategy_ret, factors], axis=1, join='inner').dropna()
    if len(merged) < 12:
        return None, None
    y = merged.iloc[:, 0]
    X = merged.iloc[:, 1:]
    X = sm.add_constant(X)
    if nw_lags is None:
        nw_lags = int(4 * (len(y) / 100) ** (2/9))
    try:
        nw = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
        ols = sm.OLS(y, X).fit()
    except Exception:
        return None, None
    res = {
        'alpha_monthly': nw.params['const'],
        'alpha_annual': nw.params['const'] * 12,
        't_stat': nw.tvalues['const'],
        'p_value': nw.pvalues['const'],
        'R2': ols.rsquared,
        'n_obs': len(y),
    }
    for f in factors.columns:
        res[f'{f}_beta'] = nw.params.get(f, np.nan)
        res[f'{f}_t'] = nw.tvalues.get(f, np.nan)
    resid = pd.Series(ols.resid, index=merged.index)
    return res, resid

def grs_test(strategy_rets, factors, residuals_dict):
    common_idx = strategy_rets.index.intersection(factors.index)
    if len(common_idx) < 12:
        return None
    R = strategy_rets.loc[common_idx]
    F = factors.loc[common_idx]
    T = R.shape[0]
    N = R.shape[1]
    K = F.shape[1]
    if T <= N + K:
        return None
    X = sm.add_constant(F)
    alphas = []
    resid_list = []
    for col in R.columns:
        if col in residuals_dict and not residuals_dict[col].empty:
            resid_df = residuals_dict[col].loc[common_idx]
            resid_list.append(resid_df.values)
            model = sm.OLS(R[col], X).fit()
            alphas.append(model.params['const'])
        else:
            model = sm.OLS(R[col], X).fit()
            alphas.append(model.params['const'])
            resid_list.append(model.resid.values)
    alpha_vec = np.array(alphas).reshape(-1,1)
    residuals = np.array(resid_list).T
    Sigma = np.cov(residuals, rowvar=False)
    try:
        inv_Sigma = np.linalg.inv(Sigma)
    except:
        inv_Sigma = np.linalg.pinv(Sigma)
    Omega = F.cov().values
    mu_f = F.mean().values.reshape(-1,1)
    inv_Omega = np.linalg.pinv(Omega)
    quad = alpha_vec.T @ inv_Sigma @ alpha_vec
    denom = 1 + mu_f.T @ inv_Omega @ mu_f
    grs = (T - N - K) / N * (quad / denom)
    grs_val = float(grs)
    p_val = float(1 - f_dist.cdf(grs_val, N, T - N - K))
    return {'GRS': grs_val, 'p_value': p_val, 'N': N, 'T': T, 'K': K}

def extract_feature_string(section_name):
    parts = section_name.split('_')
    if len(parts) > 1 and parts[0].startswith('Sec'):
        feat_parts = parts[1:]
    else:
        feat_parts = parts
    display_parts = [FEATURE_MAP.get(f, f) for f in feat_parts]
    return ', '.join(display_parts)

def load_summary_metrics(strategy_key):
    path_map = {
        'APTree_K20': PROJECT_ROOT / 'output' / 'pruned' / 'AP-Tree_full' / 'All_Sections_Summary_kmax20.csv',
        'APTree_K40': PROJECT_ROOT / 'output' / 'pruned' / 'AP-Tree_full' / 'All_Sections_Summary_kmax40.csv',
        'APTree_clean_K20': PROJECT_ROOT / 'output' / 'pruned' / 'AP-Tree_cleaned' / 'All_Sections_Summary_kmax20.csv',
        'APTree_clean_K40': PROJECT_ROOT / 'output' / 'pruned' / 'AP-Tree_cleaned' / 'All_Sections_Summary_kmax40.csv',
        'TripleSort64': PROJECT_ROOT / 'output' / 'pruned' / 'TripleSort64_full' / 'All_TripleSort_Sharpe_Summary.csv',
        'TripleSort128_clean': PROJECT_ROOT / 'output' / 'pruned' / 'TripleSort128_cleaned' / 'All_TripleSort_Sharpe_Summary.csv',
        'TripleSort128_longonly': PROJECT_ROOT / 'output' / 'pruned' / 'TripleSort128_cleaned_longonly' / 'Summary_TripleSort128_longonly.csv',
        'APTree_longonly_K5': PROJECT_ROOT / 'output' / 'pruned' / 'AP-Tree_cleaned_longonly' / 'Summary_APT_longonly.csv',
    }
    if strategy_key not in path_map:
        return {}
    path = path_map[strategy_key]
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding='utf-8-sig')
    sec_col = 'Section' if 'Section' in df.columns else '截面名称' if '截面名称' in df.columns else None
    if sec_col is None:
        return {}
    test_col = 'Test_Sharpe' if 'Test_Sharpe' in df.columns else 'Test_SR' if 'Test_SR' in df.columns else '测试集月度夏普'
    train_col = 'Train_Sharpe' if 'Train_Sharpe' in df.columns else 'Train_SR' if 'Train_SR' in df.columns else '训练集月度夏普'
    cv_col = 'CV_Sharpe' if 'CV_Sharpe' in df.columns else 'CV_SR' if 'CV_SR' in df.columns else 'CV_夏普'
    lam0_col = 'Best_λ0' if 'Best_λ0' in df.columns else '最优λ0'
    lam2_col = 'Best_λ2' if 'Best_λ2' in df.columns else '最优λ2'
    summary = {}
    for _, row in df.iterrows():
        sec = row[sec_col]
        summary[sec] = {
            'test_sharpe': row[test_col] if test_col in row else np.nan,
            'train_sharpe': row[train_col] if train_col in row else np.nan,
            'cv_sharpe': row[cv_col] if cv_col in row else np.nan,
            'lam0': row[lam0_col] if lam0_col in row else np.nan,
            'lam2': row[lam2_col] if lam2_col in row else np.nan,
        }
    return summary

def main():
    print("=" * 80)
    print("Factor Model Tests for All Portfolio Strategies")
    print("=" * 80)
    out_dir = Path(CONFIG['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    factor_models = {}
    for name in CONFIG['models']:
        print(f"Loading {name} factors...")
        factor_models[name] = load_factors(name)

    results_store = {}
    grs_store = {}

    for strat_key, rel_path in CONFIG['strategies'].items():
        file_path = PROJECT_ROOT / rel_path
        if not file_path.exists():
            print(f"Warning: {strat_key} file not found: {file_path}")
            continue
        print(f"\nProcessing {strat_key}...")
        ret_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        ret_df.index = ret_df.index.strftime('%Y-%m')
        ret_df = ret_df.sort_index()
        print(f"  Loaded {ret_df.shape[1]} portfolios")

        results_store[strat_key] = {}
        for model_name, factors in factor_models.items():
            common = ret_df.index.intersection(factors.index)
            if len(common) < 12:
                print(f"  Model {model_name}: insufficient overlap ({len(common)} months), skip")
                continue
            model_res = []
            residuals = {}
            for col in ret_df.columns:
                ret_ser = ret_df[col].dropna()
                if len(ret_ser) < 12:
                    continue
                res, resid = run_factor_regression(ret_ser, factors, CONFIG['nw_lags'])
                if res is not None:
                    res['portfolio'] = col
                    model_res.append(res)
                    residuals[col] = resid
            if model_res:
                df_res = pd.DataFrame(model_res).set_index('portfolio')
                results_store[strat_key][model_name] = df_res
                if len(df_res) >= 2:
                    common_rets = ret_df[df_res.index].loc[common]
                    grs_out = grs_test(common_rets, factors, residuals)
                    if grs_out is not None:
                        grs_store[(strat_key, model_name)] = grs_out
                print(f"  Model {model_name}: {len(df_res)} portfolios, GRS={'OK' if (strat_key,model_name) in grs_store else 'skip'}")

    # ---- 1. APTree GRS original ----
    grs_original = []
    for kmax in [20, 40]:
        strat_key = f'APTree_K{kmax}'
        for model in ['CH3', 'CH4', 'Carhart4', 'FF5', 'FF6']:
            if (strat_key, model) in grs_store:
                g = grs_store[(strat_key, model)]
                grs_original.append({
                    'Model': model,
                    'K': kmax,
                    'N': g['N'],
                    'df': f"({g['N']},{g['T']-g['N']-g['K']})",
                    'GRS_statistic': f"{g['GRS']:.3f}",
                    'p_value': f"{g['p_value']:.4f}"
                })
    if grs_original:
        pd.DataFrame(grs_original).to_csv(out_dir / 'APTree_GRS_original.csv', index=False)
        print("Saved APTree_GRS_original.csv")
    else:
        print("No GRS data for AP-Tree original sample.")

    # ---- 2. TripleSort64 top10 CH4 alpha ----
    strat_key = 'TripleSort64'
    if strat_key in results_store and 'CH4' in results_store[strat_key]:
        df_ch4 = results_store[strat_key]['CH4']
        top10 = df_ch4['alpha_annual'].sort_values(ascending=False).head(10)
        summary_metrics = load_summary_metrics(strat_key)
        rows = []
        for port in top10.index:
            row = {'Feature': extract_feature_string(port)}
            for m in ['CH3', 'CH4', 'Carhart4', 'FF5', 'FF6']:
                if m in results_store[strat_key]:
                    alpha = results_store[strat_key][m].loc[port, 'alpha_annual'] if port in results_store[strat_key][m].index else np.nan
                    row[f'alpha_{m}'] = f"{alpha*100:.2f}%" if not pd.isna(alpha) else ''
            if port in summary_metrics:
                row['Test_Sharpe'] = f"{summary_metrics[port]['test_sharpe']:.4f}" if not pd.isna(summary_metrics[port]['test_sharpe']) else ''
                row['Train_Sharpe'] = f"{summary_metrics[port]['train_sharpe']:.4f}" if not pd.isna(summary_metrics[port]['train_sharpe']) else ''
                row['CV_Sharpe'] = f"{summary_metrics[port]['cv_sharpe']:.4f}" if not pd.isna(summary_metrics[port]['cv_sharpe']) else ''
                row['lambda'] = f"({summary_metrics[port]['lam0']}, {summary_metrics[port]['lam2']})" if not pd.isna(summary_metrics[port]['lam0']) else ''
            else:
                row['Test_Sharpe'] = row['Train_Sharpe'] = row['CV_Sharpe'] = row['lambda'] = ''
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / 'TripleSort64_top10_CH4_alpha.csv', index=False)
            print("Saved TripleSort64_top10_CH4_alpha.csv")
    else:
        print("TripleSort64 CH4 results missing.")

    # ---- 3. APTree K20 top10 CH4 alpha ----
    strat_key = 'APTree_K20'
    if strat_key in results_store and 'CH4' in results_store[strat_key]:
        df_ch4 = results_store[strat_key]['CH4']
        top10 = df_ch4['alpha_annual'].sort_values(ascending=False).head(10)
        summary_metrics = load_summary_metrics(strat_key)
        rows = []
        for port in top10.index:
            row = {'Feature': extract_feature_string(port)}
            for m in ['CH3', 'CH4', 'Carhart4', 'FF5', 'FF6']:
                if m in results_store[strat_key]:
                    alpha = results_store[strat_key][m].loc[port, 'alpha_annual'] if port in results_store[strat_key][m].index else np.nan
                    row[f'alpha_{m}'] = f"{alpha*100:.2f}%" if not pd.isna(alpha) else ''
            if port in summary_metrics:
                row['Test_Sharpe'] = f"{summary_metrics[port]['test_sharpe']:.4f}" if not pd.isna(summary_metrics[port]['test_sharpe']) else ''
                row['Train_Sharpe'] = f"{summary_metrics[port]['train_sharpe']:.4f}" if not pd.isna(summary_metrics[port]['train_sharpe']) else ''
                row['CV_Sharpe'] = f"{summary_metrics[port]['cv_sharpe']:.4f}" if not pd.isna(summary_metrics[port]['cv_sharpe']) else ''
                row['lambda'] = f"({summary_metrics[port]['lam0']}, {summary_metrics[port]['lam2']})" if not pd.isna(summary_metrics[port]['lam0']) else ''
            else:
                row['Test_Sharpe'] = row['Train_Sharpe'] = row['CV_Sharpe'] = row['lambda'] = ''
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / 'APTree_K20_top10_CH4_alpha.csv', index=False)
            print("Saved APTree_K20_top10_CH4_alpha.csv")
    else:
        print("APTree_K20 CH4 results missing.")

    # ---- 4. APTree K40 top10 CH4 alpha ----
    strat_key = 'APTree_K40'
    if strat_key in results_store and 'CH4' in results_store[strat_key]:
        df_ch4 = results_store[strat_key]['CH4']
        top10 = df_ch4['alpha_annual'].sort_values(ascending=False).head(10)
        summary_metrics = load_summary_metrics(strat_key)
        rows = []
        for port in top10.index:
            row = {'Feature': extract_feature_string(port)}
            for m in ['CH3', 'CH4', 'Carhart4', 'FF5', 'FF6']:
                if m in results_store[strat_key]:
                    alpha = results_store[strat_key][m].loc[port, 'alpha_annual'] if port in results_store[strat_key][m].index else np.nan
                    row[f'alpha_{m}'] = f"{alpha*100:.2f}%" if not pd.isna(alpha) else ''
            if port in summary_metrics:
                row['Test_Sharpe'] = f"{summary_metrics[port]['test_sharpe']:.4f}" if not pd.isna(summary_metrics[port]['test_sharpe']) else ''
                row['Train_Sharpe'] = f"{summary_metrics[port]['train_sharpe']:.4f}" if not pd.isna(summary_metrics[port]['train_sharpe']) else ''
                row['CV_Sharpe'] = f"{summary_metrics[port]['cv_sharpe']:.4f}" if not pd.isna(summary_metrics[port]['cv_sharpe']) else ''
                row['lambda'] = f"({summary_metrics[port]['lam0']}, {summary_metrics[port]['lam2']})" if not pd.isna(summary_metrics[port]['lam0']) else ''
            else:
                row['Test_Sharpe'] = row['Train_Sharpe'] = row['CV_Sharpe'] = row['lambda'] = ''
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / 'APTree_K40_top10_CH4_alpha.csv', index=False)
            print("Saved APTree_K40_top10_CH4_alpha.csv")
    else:
        print("APTree_K40 CH4 results missing.")

    # ---- 5. APTree clean K20 top8 CH4 alpha ----
    strat_key = 'APTree_clean_K20'
    if strat_key in results_store and 'CH4' in results_store[strat_key]:
        df_ch4 = results_store[strat_key]['CH4']
        top8 = df_ch4['alpha_annual'].sort_values(ascending=False).head(8)
        summary_metrics = load_summary_metrics(strat_key)
        rows = []
        for port in top8.index:
            row = {'Feature': extract_feature_string(port)}
            for m in ['CH3', 'CH4', 'Carhart4', 'FF5', 'FF6']:
                if m in results_store[strat_key]:
                    alpha = results_store[strat_key][m].loc[port, 'alpha_annual'] if port in results_store[strat_key][m].index else np.nan
                    row[f'alpha_{m}'] = f"{alpha*100:.2f}%" if not pd.isna(alpha) else ''
            if port in summary_metrics:
                row['Test_Sharpe'] = f"{summary_metrics[port]['test_sharpe']:.4f}" if not pd.isna(summary_metrics[port]['test_sharpe']) else ''
                row['Train_Sharpe'] = f"{summary_metrics[port]['train_sharpe']:.4f}" if not pd.isna(summary_metrics[port]['train_sharpe']) else ''
                row['CV_Sharpe'] = f"{summary_metrics[port]['cv_sharpe']:.4f}" if not pd.isna(summary_metrics[port]['cv_sharpe']) else ''
                row['lambda'] = f"({summary_metrics[port]['lam0']}, {summary_metrics[port]['lam2']})" if not pd.isna(summary_metrics[port]['lam0']) else ''
            else:
                row['Test_Sharpe'] = row['Train_Sharpe'] = row['CV_Sharpe'] = row['lambda'] = ''
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / 'APTree_clean_K20_top8_alpha.csv', index=False)
            print("Saved APTree_clean_K20_top8_alpha.csv")
    else:
        print("APTree_clean_K20 CH4 results missing.")

    # ---- 6. APTree clean K40 top8 CH4 alpha ----
    strat_key = 'APTree_clean_K40'
    if strat_key in results_store and 'CH4' in results_store[strat_key]:
        df_ch4 = results_store[strat_key]['CH4']
        top8 = df_ch4['alpha_annual'].sort_values(ascending=False).head(8)
        summary_metrics = load_summary_metrics(strat_key)
        rows = []
        for port in top8.index:
            row = {'Feature': extract_feature_string(port)}
            for m in ['CH3', 'CH4', 'Carhart4', 'FF5', 'FF6']:
                if m in results_store[strat_key]:
                    alpha = results_store[strat_key][m].loc[port, 'alpha_annual'] if port in results_store[strat_key][m].index else np.nan
                    row[f'alpha_{m}'] = f"{alpha*100:.2f}%" if not pd.isna(alpha) else ''
            if port in summary_metrics:
                row['Test_Sharpe'] = f"{summary_metrics[port]['test_sharpe']:.4f}" if not pd.isna(summary_metrics[port]['test_sharpe']) else ''
                row['Train_Sharpe'] = f"{summary_metrics[port]['train_sharpe']:.4f}" if not pd.isna(summary_metrics[port]['train_sharpe']) else ''
                row['CV_Sharpe'] = f"{summary_metrics[port]['cv_sharpe']:.4f}" if not pd.isna(summary_metrics[port]['cv_sharpe']) else ''
                row['lambda'] = f"({summary_metrics[port]['lam0']}, {summary_metrics[port]['lam2']})" if not pd.isna(summary_metrics[port]['lam0']) else ''
            else:
                row['Test_Sharpe'] = row['Train_Sharpe'] = row['CV_Sharpe'] = row['lambda'] = ''
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / 'APTree_clean_K40_top8_alpha.csv', index=False)
            print("Saved APTree_clean_K40_top8_alpha.csv")
    else:
        print("APTree_clean_K40 CH4 results missing.")

    # ---- 7. GRS cleaned methods ----
    grs_clean_rows = []
    for method_key, pretty_name in [('TripleSort128_clean', 'TripleSort128_clean'),
                                    ('APTree_clean_K20', 'APTree_clean_K20'),
                                    ('APTree_clean_K40', 'APTree_clean_K40')]:
        for model in ['CH4', 'FF5', 'FF6']:
            if (method_key, model) in grs_store:
                g = grs_store[(method_key, model)]
                grs_clean_rows.append({
                    'Method': pretty_name,
                    'Model': model,
                    'GRS_statistic': f"{g['GRS']:.3f}",
                    'p_value': f"{g['p_value']:.4f}"
                })
    if grs_clean_rows:
        pd.DataFrame(grs_clean_rows).to_csv(out_dir / 'GRS_Cleaned_Methods.csv', index=False)
        print("Saved GRS_Cleaned_Methods.csv")
    else:
        print("No GRS data for cleaned sample methods.")

    # ---- 8. All detailed alpha results (for later use) ----
    all_alpha = []
    for strat_key, models in results_store.items():
        for model_name, df_res in models.items():
            for port in df_res.index:
                all_alpha.append({
                    'Strategy': strat_key,
                    'Model': model_name,
                    'Portfolio': port,
                    'alpha_annual': df_res.loc[port, 'alpha_annual'],
                    't_stat': df_res.loc[port, 't_stat'],
                    'p_value': df_res.loc[port, 'p_value'],
                    'R2': df_res.loc[port, 'R2']
                })
    if all_alpha:
        df_all_alpha = pd.DataFrame(all_alpha)
        df_all_alpha.to_csv(out_dir / 'All_Alpha_Detailed.csv', index=False)
        print("Saved All_Alpha_Detailed.csv")
    else:
        print("No detailed alpha results to save.")

        # ---- 9. Long-only comparison built from All_Alpha_Detailed.csv (using extract_feature_string) ----
    alpha_file = out_dir / 'All_Alpha_Detailed.csv'
    if alpha_file.exists():
        df_alpha = pd.read_csv(alpha_file)
        longonly_strategies = ['APTree_longonly_K5', 'TripleSort128_longonly']
        df_lo = df_alpha[(df_alpha['Strategy'].isin(longonly_strategies)) & (df_alpha['Model'] == 'CH4')]
        
        if not df_lo.empty:
            apt_dict = {}
            ts_dict = {}
            
            for _, row in df_lo.iterrows():
                port = row['Portfolio']
                # Remove '_triplesort' suffix for TripleSort128
                if row['Strategy'] == 'TripleSort128_longonly' and port.endswith('_triplesort'):
                    port = port[:-11]
                display_feat = extract_feature_string(port)
                alpha_val = row['alpha_annual']
                p_val = row['p_value']
                if row['Strategy'] == 'APTree_longonly_K5':
                    apt_dict[display_feat] = (alpha_val, p_val)
                else:
                    ts_dict[display_feat] = (alpha_val, p_val)
            
            all_features = sorted(set(apt_dict.keys()) | set(ts_dict.keys()))
            rows_lo = []
            for feat in all_features:
                row = {'Feature': feat}
                # AP-Tree
                if feat in apt_dict:
                    alpha, p = apt_dict[feat]
                    row['APT_Alpha'] = f"{alpha*100:.2f}%" if not pd.isna(alpha) else ''
                    row['APT_p'] = f"{p:.3f}" if not pd.isna(p) else ''
                else:
                    row['APT_Alpha'] = 'no data'
                    row['APT_p'] = 'no data'
                # TripleSort128
                if feat in ts_dict:
                    alpha, p = ts_dict[feat]
                    row['TS_Alpha'] = f"{alpha*100:.2f}%" if not pd.isna(alpha) else ''
                    row['TS_p'] = f"{p:.3f}" if not pd.isna(p) else ''
                else:
                    row['TS_Alpha'] = 'no data'
                    row['TS_p'] = 'no data'
                rows_lo.append(row)
            
            # Averages (only over rows with actual data)
            apt_alphas = [float(r['APT_Alpha'].replace('%','')) for r in rows_lo if r['APT_Alpha'] not in ('', 'no data')]
            ts_alphas = [float(r['TS_Alpha'].replace('%','')) for r in rows_lo if r['TS_Alpha'] not in ('', 'no data')]
            avg_row = {'Feature': 'Average'}
            avg_row['APT_Alpha'] = f"{np.mean(apt_alphas):.2f}%" if apt_alphas else 'no data'
            avg_row['APT_p'] = '—'
            avg_row['TS_Alpha'] = f"{np.mean(ts_alphas):.2f}%" if ts_alphas else 'no data'
            avg_row['TS_p'] = '—'
            rows_lo.append(avg_row)
            
            df_lo_out = pd.DataFrame(rows_lo)
            df_lo_out.to_csv(out_dir / 'LongOnly_Comparison.csv', index=False, encoding='utf-8-sig')
            print("Saved LongOnly_Comparison.csv (using extract_feature_string)")
        else:
            print("No long-only CH4 data found in All_Alpha_Detailed.csv.")
    else:
        print("All_Alpha_Detailed.csv not found.")
        
if __name__ == "__main__":
    main()