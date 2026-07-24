"""
TripleSort64 (full sample) - LARS pruning
Read output/candidate_pools/TripleSort64_full/Sec*_triplesort.csv
Prune each cross-section using a fixed train/valid time-series split (2:1),
output test-period metrics:
  - monthly average excess return
  - maximum drawdown
  - annualised Sharpe ratio
Output: output/pruned/TripleSort64_full/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
from sklearn.linear_model import lars_path

warnings.filterwarnings('ignore')

# ==================== Dynamic path config ====================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CONFIG = {
    'triplesort_dir': str(PROJECT_ROOT / 'output' / 'candidate_pools' / 'TripleSort64_full'),
    'output_dir': str(PROJECT_ROOT / 'output' / 'pruned' / 'TripleSort64_full'),
    'train_end_date': '2020-12-31',
    'test_start_date': '2021-01-01',
    'valid_ratio': 1/3,              # fraction of training period used for validation (last portion)
    'lambda0_list': [
        0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
    ],
    'lambda2_list': [
        0.1**5.00, 0.1**5.25, 0.1**5.50, 0.1**5.75, 0.1**6.00,
        0.1**6.25, 0.1**6.50, 0.1**6.75, 0.1**7.00, 0.1**7.25,
        0.1**7.50, 0.1**7.75, 0.1**8.00,
    ],
    'kmin': 5,
    'kmax': 16,
}

def robust_mean_estimate(mu, lambda0):
    mu_bar = np.mean(mu)
    return mu + lambda0 * mu_bar

def robust_covariance_estimate(sigma, lambda2, n_assets):
    return sigma + lambda2 * np.eye(n_assets)

def gls_transform_for_lasso(sigma_robust, mu_robust):
    try:
        L = np.linalg.cholesky(sigma_robust)
        X = L.T
        y = np.linalg.solve(L, mu_robust)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(sigma_robust)
        eigvals = np.maximum(eigvals, 1e-12)
        sqrt_sigma = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        inv_sqrt_sigma = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        X = sqrt_sigma
        y = inv_sqrt_sigma @ mu_robust
    return X, y

def lars_path_auto(X, y, kmin, kmax, max_iter=100):
    n, p = X.shape
    try:
        alphas, active, coefs = lars_path(X, y, method='lasso', max_iter=max_iter)
        beta_path = coefs.T
        K = np.sum(np.abs(beta_path) > 1e-12, axis=1)
        mask = (K >= kmin) & (K <= kmax)
        if np.any(mask):
            beta_filtered = beta_path[mask]
            K_filtered = K[mask].astype(int)
            sort_idx = np.argsort(K_filtered)
            return beta_filtered[sort_idx], K_filtered[sort_idx]
        if len(K) > 0:
            idx_kmin = np.argmin(np.abs(K - kmin))
            idx_kmax = np.argmin(np.abs(K - kmax))
            if idx_kmin == idx_kmax:
                return beta_path[[idx_kmin]], K[[idx_kmin]].astype(int)
            else:
                return beta_path[[idx_kmin, idx_kmax]], K[[idx_kmin, idx_kmax]].astype(int)
        return np.empty((0, p)), np.array([], dtype=int)
    except Exception as e:
        print(f"LARS path computation failed: {e}")
        return np.empty((0, p)), np.array([], dtype=int)

def solve_sparse_sdf_via_lars(returns_train, lambda0, lambda2, kmin, kmax, depth_weights=None):
    n_samples, n_assets = returns_train.shape
    mu = np.mean(returns_train, axis=0)
    sigma = np.cov(returns_train, rowvar=False, bias=True)
    if depth_weights is not None:
        returns_weighted = returns_train / depth_weights
        mu = np.mean(returns_weighted, axis=0)
        sigma = np.cov(returns_weighted, rowvar=False, bias=True)
    mu_robust = robust_mean_estimate(mu, lambda0)
    sigma_robust = robust_covariance_estimate(sigma, lambda2, n_assets)
    sigma_robust += 1e-8 * np.eye(n_assets)
    X, y = gls_transform_for_lasso(sigma_robust, mu_robust)
    beta_path, K_path = lars_path_auto(X, y, kmin, kmax)
    results = []
    for beta, K in zip(beta_path, K_path):
        if depth_weights is not None:
            weights = beta / depth_weights
        else:
            weights = beta.copy()
        abs_sum = np.sum(np.abs(weights))
        if abs_sum > 1e-12:
            weights = weights / abs_sum
        results.append({
            'weights': weights,
            'K': K,
            'lambda0': lambda0,
            'lambda2': lambda2,
        })
    return results

def evaluate_portfolio(returns, weights):
    if len(weights) == 0 or np.sum(np.abs(weights)) < 1e-12:
        return -np.inf
    port_returns = returns @ weights
    port_returns = port_returns[~np.isnan(port_returns)]
    if len(port_returns) < 2:
        return -np.inf
    mean_ret = np.mean(port_returns)
    std_ret = np.std(port_returns, ddof=1)
    if std_ret < 1e-12:
        return -np.inf
    return mean_ret / std_ret

def prune_single_section(df, section_name, config):
    print(f"\n  Processing section: {section_name}")

    node_names = df.columns.tolist()
    excess_df = df.fillna(0.0)

    train_mask = df.index <= pd.to_datetime(config['train_end_date'])
    test_mask = df.index >= pd.to_datetime(config['test_start_date'])

    X_train = excess_df[train_mask].values.astype(np.float64)
    X_test = excess_df[test_mask].values.astype(np.float64)

    if X_train.shape[0] < 12:
        print(f"    Warning: insufficient training samples ({X_train.shape[0]} periods), skip")
        return None

    # ---- time-series train/validation split (fixed order, 2:1) -------------
    n_train_total = X_train.shape[0]
    n_valid = max(1, int(np.floor(n_train_total * config['valid_ratio'])))
    n_train_sub = n_train_total - n_valid
    X_tr = X_train[:n_train_sub]
    X_val = X_train[n_train_sub:]       # most recent part
    if X_val.shape[0] == 0:
        X_val = X_tr[-1:].copy()

    depth_weights = None

    cv_results = []
    for lam0 in config['lambda0_list']:
        for lam2 in config['lambda2_list']:
            solutions = solve_sparse_sdf_via_lars(
                X_tr, lam0, lam2,
                config['kmin'], config['kmax'],
                depth_weights
            )
            if not solutions:
                continue
            best_val_sr = -np.inf
            for sol in solutions:
                val_sr = evaluate_portfolio(X_val, sol['weights'])
                if val_sr > best_val_sr:
                    best_val_sr = val_sr
            if best_val_sr > -np.inf:
                cv_results.append({
                    'lam0': lam0,
                    'lam2': lam2,
                    'val_sr': best_val_sr,
                })

    if not cv_results:
        print(f"    Validation failed, using default parameters")
        best_lam0, best_lam2 = 0.2, 0.001
        best_val_sr = 0.0
    else:
        best = max(cv_results, key=lambda x: x['val_sr'])
        best_lam0, best_lam2 = best['lam0'], best['lam2']
        best_val_sr = best['val_sr']
        print(f"    Optimal hyperparameters: λ0={best_lam0}, λ2={best_lam2}, Val SR={best_val_sr:.4f}")

    all_solutions = solve_sparse_sdf_via_lars(
        X_train, best_lam0, best_lam2,
        config['kmin'], config['kmax'],
        depth_weights
    )
    if not all_solutions:
        print(f"    Final training no solution")
        return None

    best_solution = None
    best_train_sr = -np.inf
    for sol in all_solutions:
        tr_sr = evaluate_portfolio(X_train, sol['weights'])
        if tr_sr > best_train_sr:
            best_train_sr = tr_sr
            best_solution = sol

    if best_solution is None:
        return None

    final_weights = best_solution['weights']
    final_K = best_solution['K']

    # ---- test-set metrics ---------------------------------------------
    if len(X_test) > 0:
        test_excess_series = X_test @ final_weights
        test_dates = df.index[test_mask]

        # Monthly Sharpe
        test_sr_monthly = evaluate_portfolio(X_test, final_weights)

        # Monthly average excess return
        monthly_avg_excess = np.mean(test_excess_series)

        # Maximum drawdown (based on excess returns cumulative)
        cum_ret = (1 + test_excess_series).cumprod()
        running_max = np.maximum.accumulate(cum_ret)
        drawdown = (cum_ret - running_max) / running_max
        max_drawdown = drawdown.min()

        # Annualised Sharpe
        annualised_sharpe = test_sr_monthly * np.sqrt(12) if not np.isnan(test_sr_monthly) else np.nan

        # Test excess return series
        test_returns_series = pd.Series(
            test_excess_series,
            index=test_dates,
            name=section_name
        )
    else:
        test_sr_monthly = np.nan
        monthly_avg_excess = np.nan
        max_drawdown = np.nan
        annualised_sharpe = np.nan
        test_returns_series = pd.Series(dtype=float)

    selected_nodes = []
    for name, w in zip(node_names, final_weights):
        if np.abs(w) > 1e-8:
            selected_nodes.append({'node_name': name, 'weight': w})
    selected_K = len(selected_nodes)

    print(f"    Done: {len(node_names)} → {selected_K} portfolios, "
          f"test monthly Sharpe = {test_sr_monthly:.4f}" if not np.isnan(test_sr_monthly)
          else "    Test Sharpe: no data")

    return {
        'section_name': section_name,
        'train_sharpe': best_train_sr,
        'val_sharpe': best_val_sr,
        'test_sharpe_monthly': test_sr_monthly,
        'monthly_avg_excess_return': monthly_avg_excess,
        'max_drawdown': max_drawdown,
        'annualised_sharpe': annualised_sharpe,
        'best_lam0': best_lam0,
        'best_lam2': best_lam2,
        'n_assets': len(node_names),
        'selected_K': selected_K,
        'selected_nodes': selected_nodes,
        'test_returns': test_returns_series,
    }

if __name__ == "__main__":
    print("=" * 120)
    print("=== TripleSort64 (full sample) - LARS pruning with fixed time-series validation ===")
    print(f"Training end: {CONFIG['train_end_date']}, Test start: {CONFIG['test_start_date']}")
    print("=== Fixed train/valid split (2:1), output monthly avg excess, max DD, annual SR ===")
    print("=" * 120)

    triplesort_dir = Path(CONFIG['triplesort_dir'])
    if not triplesort_dir.exists():
        print(f"Error: candidate pool directory not found: {triplesort_dir}")
        print("Please run build_pools.py first to generate candidate pools")
        sys.exit(1)

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    section_dfs = {}
    for file in sorted(triplesort_dir.glob("Sec*_triplesort.csv")):
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        section_dfs[file.stem] = df
        print(f"Loaded: {file.name} ({df.shape[1]} portfolios)")

    print(f"\nLoaded {len(section_dfs)} cross-sections, starting pruning...\n")

    all_results = []
    for sec_name, df in section_dfs.items():
        result = prune_single_section(df, sec_name, CONFIG)
        if result:
            all_results.append(result)

    print(f"\nSuccessfully processed {len(all_results)} cross-sections")

    if all_results:
        # Summary table
        summary_df = pd.DataFrame([{
            'Section': r['section_name'],
            'Original_Portfolios': r['n_assets'],
            'Selected_Portfolios': r['selected_K'],
            'Compression_Ratio': f"{r['selected_K']/r['n_assets']*100:.1f}%",
            'Train_Sharpe': f"{r['train_sharpe']:.4f}",
            'Val_Sharpe': f"{r['val_sharpe']:.4f}",
            'Test_Monthly_Sharpe': f"{r['test_sharpe_monthly']:.4f}",
            'Monthly_Avg_Excess_Return': f"{r['monthly_avg_excess_return']:.6f}",
            'Max_Drawdown': f"{r['max_drawdown']*100:.2f}%" if not np.isnan(r['max_drawdown']) else "NaN",
            'Annualised_Sharpe': f"{r['annualised_sharpe']:.4f}",
            'Best_λ0': r['best_lam0'],
            'Best_λ2': r['best_lam2'],
        } for r in all_results])
        summary_path = output_dir / "All_TripleSort_Sharpe_Summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\nSummary saved: {summary_path}")

        # Detailed results with selected nodes
        detailed_rows = []
        for r in all_results:
            detailed_rows.append({
                'Section': r['section_name'],
                'Type': 'Section Info',
                'Portfolio_Name': '',
                'Weight': '',
                'Train_Sharpe': f"{r['train_sharpe']:.4f}",
                'Val_Sharpe': f"{r['val_sharpe']:.4f}",
                'Test_Monthly_Sharpe': f"{r['test_sharpe_monthly']:.4f}",
                'Annualised_Sharpe': f"{r['annualised_sharpe']:.4f}",
                'Monthly_Avg_Excess_Return': f"{r['monthly_avg_excess_return']:.6f}",
                'Max_Drawdown': f"{r['max_drawdown']*100:.2f}%" if not np.isnan(r['max_drawdown']) else "NaN",
                'Best_λ0': r['best_lam0'],
                'Best_λ2': r['best_lam2'],
                'Original_Portfolios': r['n_assets'],
                'Selected_Portfolios': r['selected_K']
            })
            for node in r['selected_nodes']:
                detailed_rows.append({
                    'Section': r['section_name'],
                    'Type': 'Selected Node',
                    'Portfolio_Name': node['node_name'],
                    'Weight': f"{node['weight']:.6f}",
                    'Train_Sharpe': '',
                    'Val_Sharpe': '',
                    'Test_Monthly_Sharpe': '',
                    'Annualised_Sharpe': '',
                    'Monthly_Avg_Excess_Return': '',
                    'Max_Drawdown': '',
                    'Best_λ0': '',
                    'Best_λ2': '',
                    'Original_Portfolios': '',
                    'Selected_Portfolios': ''
                })
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_path = output_dir / "All_TripleSort_Detailed_Results.csv"
        detailed_df.to_csv(detailed_path, index=False, encoding='utf-8-sig')
        print(f"Detailed results saved: {detailed_path}")

        # Test excess return series
        test_returns_dict = {}
        for r in all_results:
            if 'test_returns' in r and not r['test_returns'].empty:
                test_returns_dict[r['section_name']] = r['test_returns']
        if test_returns_dict:
            test_returns_df = pd.concat(test_returns_dict, axis=1)
            test_returns_path = output_dir / "Test_Excess_Returns_TripleSort64.csv"
            test_returns_df.to_csv(test_returns_path, encoding='utf-8-sig')
            print(f"Test excess return series saved: {test_returns_path}")
    else:
        print("No valid results")

    print("\n" + "=" * 120)
    print("Processing completed!")
    print("=" * 120)
