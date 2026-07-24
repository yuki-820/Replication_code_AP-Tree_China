"""
Comprehensive factor model testing – revised output logic:
- Only portfolios with significant alpha (p<0.05) in ALL FIVE factor models are kept.
- Top 10 by test monthly Sharpe for each strategy-weight scheme.
- Separate GRS tables for full/cleaned samples.
- Long-only table: 36 sections × 3 strategies (missing if not significant in any model).
- Combined large table with all significant results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from scipy.stats import f as f_dist
import warnings
import re

warnings.filterwarnings('ignore')

# ==================== Configuration (same as before) ====================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CONFIG = {
    'factor_dir': str(PROJECT_ROOT / 'data' / 'factors'),
    'output_dir': str(PROJECT_ROOT / 'output' / 'tables'),
    'models': {
        'CH3': {'file': 'CH3_factors_monthly_202602.xlsx', 'sheet': 'Returnseries',
                'mapping': {'mktrf': 'Mkt-RF', 'SMB': 'SMB', 'VMG': 'VMG'}},
        'CH4': {'file': 'CH4_factors_monthly_202602.xlsx', 'sheet': 'Returnseries',
                'mapping': {'mktrf': 'Mkt-RF', 'SMB': 'SMB', 'VMG': 'VMG', 'PMO': 'PMO'}},
        'FF5': {'file': 'fivefactor_monthly.csv', 'sheet': None,
                'mapping': {'mkt_rf': 'Mkt-RF', 'smb': 'SMB', 'hml': 'HML', 'rmw': 'RMW', 'cma': 'CMA'}},
        'FF6': {'file': 'fivefactor_monthly.csv', 'sheet': None,
                'mapping': {'mkt_rf': 'Mkt-RF', 'smb': 'SMB', 'hml': 'HML', 'rmw': 'RMW', 'cma': 'CMA', 'umd': 'UMD'}},
        'Carhart4': {'file': 'fivefactor_monthly.csv', 'sheet': None,
                     'mapping': {'mkt_rf': 'Mkt-RF', 'smb': 'SMB', 'hml': 'HML', 'umd': 'UMD'}}
    },
    'strategies': {
        'TripleSort64': 'output/pruned/TripleSort64_full/Test_Excess_Returns_TripleSort64.csv',
        'APTree_K20': 'output/pruned/AP-Tree_full/Test_Excess_Returns_kmax20.csv',
        'APTree_K40': 'output/pruned/AP-Tree_full/Test_Excess_Returns_kmax40.csv',
        'TripleSort128_clean': 'output/pruned/TripleSort128_cleaned/Test_Excess_Returns_TripleSort128.csv',
        'APTree_clean_K20': 'output/pruned/AP-Tree_cleaned/Test_Excess_Returns_kmax20.csv',
        'APTree_clean_K40': 'output/pruned/AP-Tree_cleaned/Test_Excess_Returns_kmax40.csv',
        'TripleSort128_longonly': 'output/pruned/TripleSort128_cleaned_longonly/Test_Excess_Returns_TripleSort128_longonly.csv',
        'APTree_longonly_K5': 'output/pruned/AP-Tree_cleaned_longonly/Test_Excess_Returns_kmax5.csv',
    },
    'nw_lags': None,
    'significance_level': 0.05,
}

FEATURE_MAP = {
    'mkt_cap': 'LME', 'turnover': 'Lturnover', 'st_rev': 'ST_Rev',
    'r12_2': 'r12_2', 'lt_rev': 'LT_Rev', 'idio_vol': 'IdioVol',
    'fin_investment': 'Fin_Investment', 'fin_op': 'Fin_OP',
    'fin_ac': 'Fin_AC', 'beme': 'BEME'
}

# ---------- helper functions (unchanged except load_summary_metrics) ----------
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
        res[f'{f}_pvalue'] = nw.pvalues.get(f, np.nan)
    resid = pd.Series(ols.resid, index=merged.index)
    return res, resid

def grs_test(strategy_rets, factors, residuals_dict):
    common_idx = strategy_rets.index.intersection(factors.index)
    if len(common_idx) < 12:
        return None
    R = strategy_rets.loc[common_idx]
    F = factors.loc[common_idx]
    T, N, K = R.shape[0], R.shape[1], F.shape[1]
    if T <= N + K:
        return None
    X = sm.add_constant(F)
    alphas, resid_list = [], []
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
    except np.linalg.LinAlgError:
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
    name = re.sub(r'_dw_power\d+\.?\d*$', '', section_name)
    name = re.sub(r'_triplesort$', '', name)
    parts = name.split('_')
    if len(parts) > 1 and parts[0].startswith('Sec'):
        feat_parts = parts[1:]
    else:
        feat_parts = parts
    return ', '.join([FEATURE_MAP.get(f, f) for f in feat_parts])

def extract_weight_scheme(portfolio_name):
    match = re.search(r'_dw_power(\d+\.?\d*)$', portfolio_name)
    return f"dw_power{match.group(1)}" if match else ""

def load_summary_metrics(strategy_key):
    path_map = {
        'APTree_K20': PROJECT_ROOT / 'output' / 'pruned' / 'AP-Tree_full' / 'All_Sections_Summary_kmax20.csv',
        'APTree_K40': PROJECT_ROOT / 'output' / 'pruned' / 'AP-Tree_full' / 'All_Sections_Summary_kmax40.csv',
        'APTree_clean_K20': PROJECT_ROOT / 'output' / 'pruned' / 'AP-Tree_cleaned' / 'All_Sections_Summary_kmax20.csv',
        'APTree_clean_K40': PROJECT_ROOT / 'output' / 'pruned' / 'AP-Tree_cleaned' / 'All_Sections_Summary_kmax40.csv',
        'TripleSort64': PROJECT_ROOT / 'output' / 'pruned' / 'TripleSort64_full' / 'All_TripleSort_Sharpe_Summary.csv',
        'TripleSort128_clean': PROJECT_ROOT / 'output' / 'pruned' / 'TripleSort128_cleaned' / 'All_TripleSort_Sharpe_Summary.csv',
        'TripleSort128_longonly': PROJECT_ROOT / 'output' / 'pruned' / 'TripleSort128_cleaned_longonly' / 'Summary_TripleSort128_longonly.csv',
        'APTree_longonly_K5': PROJECT_ROOT / 'output' / 'pruned' / 'AP-Tree_cleaned_longonly' / 'All_Sections_Summary_kmax5.csv',
    }
    path = path_map.get(strategy_key)
    if not path or not path.exists():
        return {}
    df = pd.read_csv(path, encoding='utf-8-sig')
    sec_col = 'Section' if 'Section' in df.columns else None
    if sec_col is None:
        return {}
    sharpe_candidates = ['Test_Monthly_Sharpe', 'Test_Sharpe', 'Test_SR']
    test_col = next((c for c in sharpe_candidates if c in df.columns), None)
    if test_col is None:
        return {}
    weight_col = 'Weight_Scheme' if 'Weight_Scheme' in df.columns else None
    summary = {}
    for _, row in df.iterrows():
        sec = row[sec_col]
        if weight_col and not pd.isna(row.get(weight_col, '')):
            port_name = f"{sec}_{row[weight_col]}"
        else:
            port_name = sec
        summary[port_name] = float(row[test_col]) if not pd.isna(row[test_col]) else np.nan
    return summary

# ==================== Main ====================
def main():
    print("=" * 80)
    print("Factor Model Tests (All five models significant)")
    print("=" * 80)
    out_dir = Path(CONFIG['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    factor_models = {name: load_factors(name) for name in CONFIG['models']}

    # ---- 1. Run all regressions and store results + residuals ----
    results_store = {}      # results_store[strat][model] = DataFrame (index=portfolio)
    residuals_store = {}    # residuals_store[strat][model] = dict(portfolio -> series)
    test_sharpe_map = {}

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

        sharpe_dict = load_summary_metrics(strat_key)
        test_sharpe_map[strat_key] = sharpe_dict

        results_store[strat_key] = {}
        residuals_store[strat_key] = {}
        for model_name, factors in factor_models.items():
            common = ret_df.index.intersection(factors.index)
            if len(common) < 12:
                continue
            model_res = []
            model_resid = {}
            for col in ret_df.columns:
                ret_ser = ret_df[col].dropna()
                if len(ret_ser) < 12:
                    continue
                res, resid = run_factor_regression(ret_ser, factors, CONFIG['nw_lags'])
                if res is not None:
                    res['portfolio'] = col
                    model_res.append(res)
                    model_resid[col] = resid
            if model_res:
                df_res = pd.DataFrame(model_res).set_index('portfolio')
                results_store[strat_key][model_name] = df_res
                residuals_store[strat_key][model_name] = model_resid
                print(f"  Model {model_name}: {len(df_res)} portfolios")

    # ---- 2. Identify portfolios significant in ALL five models ----
    model_names = list(factor_models.keys())
    alpha_threshold = CONFIG['significance_level']

    # Gather all unique portfolios across all strategies
    all_portfolios_info = []   # list of dicts: strategy, portfolio, scheme, test_sharpe, alpha_p for each model
    for strat_key in results_store:
        # To be considered, portfolio must appear in all five models
        common_ports = None
        for m in model_names:
            if m in results_store[strat_key]:
                if common_ports is None:
                    common_ports = set(results_store[strat_key][m].index)
                else:
                    common_ports = common_ports.intersection(results_store[strat_key][m].index)
        if not common_ports:
            continue
        for port in common_ports:
            # collect p-values
            pvals = {}
            for m in model_names:
                pvals[m] = results_store[strat_key][m].loc[port, 'p_value']
            if all(pd.notna(pv) and pv < alpha_threshold for pv in pvals.values()):
                # it passes
                sharpe = test_sharpe_map.get(strat_key, {}).get(port, np.nan)
                if not np.isnan(sharpe):
                    all_portfolios_info.append({
                        'strategy': strat_key,
                        'portfolio': port,
                        'weight_scheme': extract_weight_scheme(port),
                        'test_sharpe': sharpe,
                        'alpha_pvals': pvals,
                    })

    if not all_portfolios_info:
        print("No portfolio passes all five factor models. Exiting.")
        return

    df_candidates = pd.DataFrame(all_portfolios_info)

    # ---- 3. For each strategy-weight scheme, take top 10 by test Sharpe ----
    # Build a list of (strategy, scheme) combinations present in candidates
    strategy_scheme_groups = df_candidates.groupby(['strategy', 'weight_scheme'])

    # We'll also need a mapping from (strategy, scheme) to "display name" used in outputs
    # For TripleSort, scheme is ""; for AP-Tree, scheme like "dw_power2.0"
    def make_display_name(strat, scheme):
        if scheme:
            return f"{strat}_{scheme}"
        else:
            return strat

    # ---- 3. Top 10 per strategy (merging depth-weight schemes within same K) ----
    strategy_groups = df_candidates.groupby('strategy')
    top10_tables = {}
    for strat, group in strategy_groups:
        top10 = group.nlargest(10, 'test_sharpe')
        display = strat  # e.g., "APTree_K20"
        rows = []
        for _, row in top10.iterrows():
            port = row['portfolio']
            feat = extract_feature_string(port)
            entry = {
                'Section_Feature': feat,
                'Portfolio': port,
                'Test_Monthly_Sharpe': row['test_sharpe'],
                'Weight_Scheme': row['weight_scheme'] if row['weight_scheme'] else 'none',
            }
            for m in model_names:
                if m in results_store[strat]:
                    reg = results_store[strat][m].loc[port]
                    entry[f'{m}_alpha_annual'] = reg['alpha_annual']
                    entry[f'{m}_alpha_t'] = reg['t_stat']
                    entry[f'{m}_alpha_p'] = row['alpha_pvals'][m]
                    entry[f'{m}_R2'] = reg['R2']
                    entry[f'{m}_Nobs'] = reg['n_obs']
            rows.append(entry)
        top10_tables[display] = pd.DataFrame(rows)
        top10_tables[display].to_csv(out_dir / f"Top10_{display}.csv", index=False, encoding='utf-8-sig')
        print(f"Saved Top10 for {display}")

    # ---- 4. GRS tables for full sample and cleaned sample ----
    # Define the five objects for each sample:
    full_sample_objects = [
        ('TripleSort64', ''),
        ('APTree_K20', 'dw_power0.5'),
        ('APTree_K20', 'dw_power2.0'),
        ('APTree_K40', 'dw_power0.5'),
        ('APTree_K40', 'dw_power2.0'),
    ]
    cleaned_sample_objects = [
        ('TripleSort128_clean', ''),
        ('APTree_clean_K20', 'dw_power0.5'),
        ('APTree_clean_K20', 'dw_power2.0'),
        ('APTree_clean_K40', 'dw_power0.5'),
        ('APTree_clean_K40', 'dw_power2.0'),
    ]

    def compute_grs_for_objects(objects, sample_label):
        grs_rows = []
        for strat, scheme in objects:
            if strat not in results_store:
                continue
            # collect portfolios belonging to this scheme
            if scheme:
                target_ports = [p for p in results_store[strat][model_names[0]].index if extract_weight_scheme(p) == scheme]
            else:
                target_ports = [p for p in results_store[strat][model_names[0]].index if extract_weight_scheme(p) == '']
            if len(target_ports) < 2:
                continue
            # for each factor model
            for model in model_names:
                if model not in results_store[strat]:
                    continue
                # select only portfolios that exist in this model's results (should be all)
                common = [p for p in target_ports if p in results_store[strat][model].index]
                if len(common) < 2:
                    continue
                # get return data
                ret_file = PROJECT_ROOT / CONFIG['strategies'][strat]
                ret_df = pd.read_csv(ret_file, index_col=0, parse_dates=True)
                ret_df.index = ret_df.index.strftime('%Y-%m')
                ret_sel = ret_df[common]
                factors = factor_models[model]
                resid_dict = {p: residuals_store[strat][model][p] for p in common}
                grs_out = grs_test(ret_sel, factors, resid_dict)
                if grs_out:
                    grs_rows.append({
                        'Sample': sample_label,
                        'Strategy': make_display_name(strat, scheme),
                        'Model': model,
                        'GRS': grs_out['GRS'],
                        'p_value': grs_out['p_value'],
                        'N': grs_out['N'],
                        'T': grs_out['T'],
                        'K': grs_out['K']
                    })
        return pd.DataFrame(grs_rows)

    df_grs_full = compute_grs_for_objects(full_sample_objects, 'Full')
    df_grs_cleaned = compute_grs_for_objects(cleaned_sample_objects, 'Cleaned')
    if not df_grs_full.empty:
        df_grs_full.to_csv(out_dir / 'GRS_Full_Sample.csv', index=False, encoding='utf-8-sig')
    if not df_grs_cleaned.empty:
        df_grs_cleaned.to_csv(out_dir / 'GRS_Cleaned_Sample.csv', index=False, encoding='utf-8-sig')
    print("GRS tables saved.")

    # ---- 5. Long-only section table (36 sections × 3 strategies) ----
    long_strategies = [
        ('TripleSort128_longonly', ''),
        ('APTree_longonly_K5', 'dw_power0.5'),
        ('APTree_longonly_K5', 'dw_power2.0'),
    ]
    # Get the list of sections (base names) from any of these strategies
    base_sec_set = set()
    for strat, scheme in long_strategies:
        if strat in results_store:
            for p in results_store[strat][model_names[0]].index:
                base = re.sub(r'_dw_power\d+\.?\d*$', '', p)
                base = re.sub(r'_triplesort$', '', base)
                base_sec_set.add(base)
    sections = sorted(base_sec_set)  # should be 36 sections
    long_rows = []
    for sec in sections:
        row = {'Section': sec, 'Feature': extract_feature_string(sec)}
        for strat, scheme in long_strategies:
            col_prefix = make_display_name(strat, scheme)
            # construct portfolio name: for TripleSort128_longonly, port = sec + "_triplesort"? Actually original file may have sec_triplesort
            if strat == 'TripleSort128_longonly':
                port = sec + '_triplesort'
            else:
                if scheme:
                    port = f"{sec}_{scheme}"
                else:
                    port = sec
            # check if it passes all five models
            passes = True
            for m in model_names:
                if m not in results_store.get(strat, {}) or port not in results_store[strat][m].index:
                    passes = False
                    break
                pv = results_store[strat][m].loc[port, 'p_value']
                if pd.isna(pv) or pv >= alpha_threshold:
                    passes = False
                    break
            if passes:
                # collect alpha info from CH4 or all? We'll store all five alphas
                for m in model_names:
                    reg = results_store[strat][m].loc[port]
                    row[f'{col_prefix}_{m}_alpha'] = reg['alpha_annual']
                    row[f'{col_prefix}_{m}_t'] = reg['t_stat']
                row[f'{col_prefix}_sharpe'] = test_sharpe_map.get(strat, {}).get(port, np.nan)
            else:
                # leave empty
                pass
        long_rows.append(row)

    df_long = pd.DataFrame(long_rows)
    df_long.to_csv(out_dir / 'LongOnly_Section_Alpha.csv', index=False, encoding='utf-8-sig')
    print("Long-only table saved.")

    # ---- 6. Combined large table with all significant results ----
    # Merge all Top10 tables together and add identifier columns
    combined_parts = []
    for (strat, scheme), group in strategy_scheme_groups:
        top10 = group.nlargest(10, 'test_sharpe')
        for _, row in top10.iterrows():
            entry = {
                'Strategy': strat,
                'Weight_Scheme': scheme if scheme else 'none',
                'Portfolio': row['portfolio'],
                'Feature': extract_feature_string(row['portfolio']),
                'Test_Monthly_Sharpe': row['test_sharpe'],
            }
            for m in model_names:
                reg = results_store[strat][m].loc[row['portfolio']]
                entry[f'{m}_alpha_annual'] = reg['alpha_annual']
                entry[f'{m}_alpha_t'] = reg['t_stat']
                entry[f'{m}_alpha_p'] = row['alpha_pvals'][m]
                entry[f'{m}_R2'] = reg['R2']
            combined_parts.append(entry)
    df_combined = pd.DataFrame(combined_parts)
    df_combined.to_csv(out_dir / 'All_Significant_Combined.csv', index=False, encoding='utf-8-sig')
    print("Combined table saved.")
    # ---- 7. Save all regression results for plotting purposes ----
    all_reg_rows = []
    for strat_key in results_store:
        for model_name, df_res in results_store[strat_key].items():
            factors = factor_models[model_name]          # factor DataFrame
            factor_names = factors.columns.tolist()
            for port in df_res.index:
                row = df_res.loc[port]
                w_scheme = extract_weight_scheme(port)
                feat = extract_feature_string(port)
                sharpe = test_sharpe_map.get(strat_key, {}).get(port, np.nan)
                base = {
                    'Strategy': strat_key,
                    'Weight_Scheme': w_scheme if w_scheme else 'none',
                    'Portfolio': port,
                    'Feature': feat,
                    'Test_Monthly_Sharpe': sharpe,
                    'Model': model_name,
                    'Alpha_annual': row['alpha_annual'],
                    'Alpha_t': row['t_stat'],
                    'Alpha_p': row['p_value'],
                    'R2': row['R2'],
                    'N_obs': row['n_obs'],
                }
                # Append factor loadings
                for f in factor_names:
                    base[f'{f}_beta'] = row.get(f'{f}_beta', np.nan)
                    base[f'{f}_t'] = row.get(f'{f}_t', np.nan)
                    base[f'{f}_p'] = row.get(f'{f}_pvalue', np.nan)
                all_reg_rows.append(base)

    if all_reg_rows:
        df_all_reg = pd.DataFrame(all_reg_rows)
        df_all_reg.to_csv(out_dir / 'All_Regression_Results.csv',
                          index=False, encoding='utf-8-sig')
        print("Full regression results saved to All_Regression_Results.csv")
    print("\nAll factor model tests completed.")

if __name__ == "__main__":
    main()
