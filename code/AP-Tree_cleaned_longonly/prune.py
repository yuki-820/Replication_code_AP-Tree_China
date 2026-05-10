"""
AP-Tree (cleaned sample) - LARS pruning with long-only constraint (max 5 nodes)
Read output/candidate_pools/AP-Tree_cleaned/Sec*.csv
Prune each cross-section, keep at most 5 nodes with depth weighting, then enforce non-negative weights and renormalize
Output: output/pruned/AP-Tree_cleaned_longonly/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
from sklearn.linear_model import lars_path
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CONFIG = {
    'tree_sections_dir': str(PROJECT_ROOT / 'output' / 'candidate_pools' / 'AP-Tree_cleaned'),
    'output_dir': str(PROJECT_ROOT / 'output' / 'pruned' / 'AP-Tree_cleaned_longonly'),
    'train_end_date': '2020-12-31',
    'test_start_date': '2021-01-01',
    'rf_col': 'rf_rate',
    'n_folds': 3,
    'shuffle': True,
    'random_state': 42,
    'lambda0_list': [
        0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
    ],
    'lambda2_list': [
        0.1**5.00, 0.1**5.25, 0.1**5.50, 0.1**5.75, 0.1**6.00,
        0.1**6.25, 0.1**6.50, 0.1**6.75, 0.1**7.00, 0.1**7.25,
        0.1**7.50, 0.1**7.75, 0.1**8.00,
    ],
    'kmin': 3,
    'target_kmax': 5,
    'depth_weight_power': 2.0,
}

def get_node_depth(name):
    try:
        last = name.split('_')[-1]
        return len(last) if last.isdigit() else 0
    except:
        return 0

def compute_depth_weights(node_names, power=2.0):
    depths = np.array([get_node_depth(name) for name in node_names])
    weights = 1.0 / np.sqrt(power ** depths)
    return weights, depths

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
        print(f"LARS path error: {e}")
        return np.empty((0, p)), np.array([], dtype=int)

def solve_sparse_sdf_via_lars(returns_train, lambda0, lambda2, kmin, kmax, depth_weights=None):
    n_samples, n_assets = returns_train.shape
    if depth_weights is not None:
        # Adjust returns by multiplying depth weights (shrink deep nodes)
        returns_adj = returns_train * depth_weights
    else:
        returns_adj = returns_train
    mu = np.mean(returns_adj, axis=0)
    sigma = np.cov(returns_adj, rowvar=False, bias=True)
    mu_robust = robust_mean_estimate(mu, lambda0)
    sigma_robust = robust_covariance_estimate(sigma, lambda2, n_assets)
    sigma_robust += 1e-8 * np.eye(n_assets)
    X, y = gls_transform_for_lasso(sigma_robust, mu_robust)
    beta_path, K_path = lars_path_auto(X, y, kmin, kmax)
    results = []
    for beta, K in zip(beta_path, K_path):
        if depth_weights is not None:
            # Recover portfolio weights: multiply beta by depth_weights and normalize
            weights = beta * depth_weights
        else:
            weights = beta.copy()
        abs_sum = np.sum(np.abs(weights))
        if abs_sum > 1e-12:
            weights = weights / abs_sum
        actual_K = int(np.sum(np.abs(weights) > 1e-8))
        results.append({
            'weights': weights,
            'K': actual_K,
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
    std = np.std(port_returns, ddof=1)
    if std < 1e-12:
        return -np.inf
    return np.mean(port_returns) / std

def prune_single_section(df, section_name, config):
    print(f"\n  Pruning: {section_name} (max nodes = {config['target_kmax']})")
    rf = df[config['rf_col']].values.reshape(-1, 1)
    node_names = [c for c in df.columns if c != config['rf_col']]
    node_returns = df[node_names].values
    excess_returns = node_returns - rf
    excess_returns = np.nan_to_num(excess_returns, nan=0.0)
    node_returns = np.nan_to_num(node_returns, nan=0.0)
    train_mask = df.index <= pd.to_datetime(config['train_end_date'])
    test_mask = df.index >= pd.to_datetime(config['test_start_date'])
    X_train = excess_returns[train_mask]
    X_test = excess_returns[test_mask]
    raw_test_returns = node_returns[test_mask]
    if X_train.shape[0] < 12 or X_train.shape[1] < config['kmin']:
        print("    Insufficient data")
        return None
    depth_weights, _ = compute_depth_weights(node_names, config['depth_weight_power'])
    kf = KFold(n_splits=min(config['n_folds'], X_train.shape[0]), shuffle=config['shuffle'], random_state=config['random_state'])
    cv_results = []
    for lam0 in config['lambda0_list']:
        for lam2 in config['lambda2_list']:
            fold_best_srs = []
            for tr_idx, val_idx in kf.split(X_train):
                X_tr = X_train[tr_idx]
                X_val = X_train[val_idx]
                solutions = solve_sparse_sdf_via_lars(X_tr, lam0, lam2, config['kmin'], config['target_kmax'], depth_weights)
                if not solutions:
                    continue
                best_val_sr = -np.inf
                for sol in solutions:
                    val_sr = evaluate_portfolio(X_val, sol['weights'])
                    if val_sr > best_val_sr:
                        best_val_sr = val_sr
                if best_val_sr > -np.inf:
                    fold_best_srs.append(best_val_sr)
            if fold_best_srs:
                cv_results.append({'lam0': lam0, 'lam2': lam2, 'avg_cv_sr': np.mean(fold_best_srs)})
    if not cv_results:
        best_lam0, best_lam2 = 0.2, 0.001
        best_cv_sr = 0.0
    else:
        best = max(cv_results, key=lambda x: x['avg_cv_sr'])
        best_lam0, best_lam2 = best['lam0'], best['lam2']
        best_cv_sr = best['avg_cv_sr']
        print(f"    Optimal: λ0={best_lam0}, λ2={best_lam2}, CV_SR={best_cv_sr:.4f}")
    all_solutions = solve_sparse_sdf_via_lars(X_train, best_lam0, best_lam2, config['kmin'], config['target_kmax'], depth_weights)
    if not all_solutions:
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
    # Apply long-only: clip negative weights to zero and renormalize
    raw_weights = best_solution['weights']
    long_weights = np.maximum(raw_weights, 0)
    sum_pos = np.sum(long_weights)
    if sum_pos > 1e-12:
        long_weights = long_weights / sum_pos
    else:
        long_weights = np.ones_like(long_weights) / len(long_weights)
    if len(X_test) > 0:
        test_sr = evaluate_portfolio(X_test, long_weights)
        test_returns = X_test @ long_weights
        test_dates = df.index[test_mask]
        test_returns_series = pd.Series(test_returns, index=test_dates, name='excess_return')
    else:
        test_sr = np.nan
        test_returns_series = pd.Series(dtype=float)
    selected_nodes = []
    for name, w in zip(node_names, long_weights):
        if w > 1e-8:
            selected_nodes.append({'node_name': name, 'depth': get_node_depth(name), 'weight': w})
    selected_K = len(selected_nodes)
    print(f"    Done: {len(node_names)} → {selected_K} nodes, test Sharpe={test_sr:.4f}")
    return {
        'section_name': section_name,
        'target_kmax': config['target_kmax'],
        'train_sharpe': best_train_sr,
        'test_sharpe': test_sr,
        'cv_sharpe': best_cv_sr,
        'best_lam0': best_lam0,
        'best_lam2': best_lam2,
        'n_assets': len(node_names),
        'selected_K': selected_K,
        'selected_nodes': selected_nodes,
        'test_returns': test_returns_series,
    }

if __name__ == "__main__":
    print("=" * 120)
    print("=== AP-Tree (cleaned sample) - Long-only pruning (max 5 nodes) ===")
    print("=" * 120)
    section_dir = Path(CONFIG['tree_sections_dir'])
    if not section_dir.exists():
        print(f"Error: candidate pool not found: {section_dir}")
        sys.exit(1)
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    section_dfs = {}
    for file in sorted(section_dir.glob("Sec*.csv")):
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        section_dfs[file.stem] = df
        print(f"Loaded: {file.name} ({df.shape[1]-1} nodes)")
    all_results = []
    for sec_name, df in section_dfs.items():
        res = prune_single_section(df, sec_name, CONFIG)
        if res:
            all_results.append(res)
    if all_results:
        summary_df = pd.DataFrame([{
            'Section': r['section_name'],
            'Original': r['n_assets'],
            'Selected': r['selected_K'],
            'Train_SR': r['train_sharpe'],
            'Test_SR': r['test_sharpe'],
            'CV_SR': r['cv_sharpe'],
            'Best_λ0': r['best_lam0'],
            'Best_λ2': r['best_lam2']
        } for r in all_results])
        summary_df.to_csv(output_dir / "Summary_APT_longonly.csv", index=False)
        test_returns_dict = {r['section_name']: r['test_returns'] for r in all_results if not r['test_returns'].empty}
        if test_returns_dict:
            pd.concat(test_returns_dict, axis=1).to_csv(output_dir / "Test_Excess_Returns_APT_longonly.csv")
        print("Results saved.")
    print("Done.")
