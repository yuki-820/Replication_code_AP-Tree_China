"""
AP-Tree (full sample) - LARS pruning
Read output/candidate_pools/AP-Tree_full/Sec*.csv
Prune each cross-section with cross-validation, output test period excess return series (kmax=20 and 40)
Output: output/pruned/AP-Tree_full/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
from sklearn.linear_model import lars_path
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

# ==================== Dynamic path configuration ====================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CONFIG = {
    'tree_sections_dir': str(PROJECT_ROOT / 'output' / 'candidate_pools' / 'AP-Tree_full'),
    'output_dir': str(PROJECT_ROOT / 'output' / 'pruned' / 'AP-Tree_full'),
    'train_end_date': '2020-12-31',
    'test_start_date': '2021-01-01',
    'rf_col': 'rf_rate',
    'n_folds': 3,
    'shuffle': True,
    'random_state': 42,
    'lambda0_list': [0.0, 0.1, 0.2, 0.4, 0.3, 0.5, 0.6, 0.8, 1.0],
    'lambda2_list': [0.06, 0.04, 0.05, 0.01, 0.02, 0.03],
    'kmin': 5,
    'target_kmax_list': [20, 40],
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
    std_ret = np.std(port_returns, ddof=1)
    if std_ret < 1e-12:
        return -np.inf
    return np.mean(port_returns) / std_ret

def prune_single_section(df, section_name, config, target_kmax):
    print(f"\n  Pruning section: {section_name} (target max nodes = {target_kmax})")
    rf = df[config['rf_col']].values.reshape(-1, 1)
    node_names = [col for col in df.columns if col != config['rf_col']]
    node_returns = df[node_names].values
    excess_returns = node_returns - rf
    excess_returns = np.nan_to_num(excess_returns, nan=0.0)
    node_returns = np.nan_to_num(node_returns, nan=0.0)
    train_mask = df.index <= pd.to_datetime(config['train_end_date'])
    test_mask = df.index >= pd.to_datetime(config['test_start_date'])
    X_train = excess_returns[train_mask]
    X_test = excess_returns[test_mask]
    # raw_test_returns 仅用于原始收益，但此处不再需要，已删除
    if X_train.shape[0] < 12 or X_train.shape[1] < config['kmin']:
        print(f"    Warning: insufficient training data ({X_train.shape[0]} periods, {X_train.shape[1]} nodes), skip")
        return None
    depth_weights, depths = compute_depth_weights(node_names, config['depth_weight_power'])
    kf = KFold(n_splits=min(config['n_folds'], X_train.shape[0]), 
               shuffle=config['shuffle'], random_state=config['random_state'])
    cv_results = []
    for lam0 in config['lambda0_list']:
        for lam2 in config['lambda2_list']:
            fold_best_srs = []
            for tr_idx, val_idx in kf.split(X_train):
                X_tr = X_train[tr_idx]
                X_val = X_train[val_idx]
                solutions = solve_sparse_sdf_via_lars(
                    X_tr, lam0, lam2,
                    config['kmin'], target_kmax,
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
                    fold_best_srs.append(best_val_sr)
            if fold_best_srs:
                cv_results.append({
                    'lam0': lam0,
                    'lam2': lam2,
                    'avg_cv_sr': np.mean(fold_best_srs),
                })
    if not cv_results:
        print(f"    Cross-validation failed, using default parameters")
        best_lam0, best_lam2 = 0.2, 0.001
        best_cv_sr = 0.0
    else:
        best = max(cv_results, key=lambda x: x['avg_cv_sr'])
        best_lam0, best_lam2 = best['lam0'], best['lam2']
        best_cv_sr = best['avg_cv_sr']
        print(f"    Optimal hyperparameters: λ0={best_lam0}, λ2={best_lam2}, CV_SR={best_cv_sr:.4f}")
    all_solutions = solve_sparse_sdf_via_lars(
        X_train, best_lam0, best_lam2,
        config['kmin'], target_kmax,
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
    if len(X_test) > 0:
        test_sr = evaluate_portfolio(X_test, final_weights)
        # 修正：使用超额收益 X_test 计算组合超额收益序列
        test_excess_returns = X_test @ final_weights
        test_dates = df.index[test_mask]
        test_returns_series = pd.Series(test_excess_returns, index=test_dates, name='excess_return')
    else:
        test_sr = np.nan
        test_returns_series = pd.Series(dtype=float)
    selected_nodes = []
    for name, w in zip(node_names, final_weights):
        if np.abs(w) > 1e-8:
            selected_nodes.append({
                'node_name': name,
                'depth': get_node_depth(name),
                'weight': w
            })
    selected_K = len(selected_nodes)
    print(f"    Done: {len(node_names)} → {selected_K} nodes, test monthly Sharpe = {test_sr:.4f}" if not np.isnan(test_sr) else "    Test Sharpe: no data")
    return {
        'section_name': section_name,
        'target_kmax': target_kmax,
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
    print("=== AP-Tree (full sample) - LARS Pruning ===")
    print(f"Training end: {CONFIG['train_end_date']}, Test start: {CONFIG['test_start_date']}")
    print("=== Allows negative weights, absolute sum normalization, output monthly Sharpe and test excess returns ===")
    print("=" * 120)

    section_dir = Path(CONFIG['tree_sections_dir'])
    if not section_dir.exists():
        print(f"Error: candidate pool directory does not exist: {section_dir}")
        print("Please run build_pools.py first to generate candidate pools")
        sys.exit(1)

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    section_dfs = {}
    for file in sorted(section_dir.glob("Sec*.csv")):
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        section_dfs[file.stem] = df
        print(f"Loaded: {file.name} ({df.shape[1]-1} nodes)")

    print(f"\nLoaded {len(section_dfs)} cross-sections, starting pruning...\n")

    all_results = []
    for sec_name, df in section_dfs.items():
        for kmax in CONFIG['target_kmax_list']:
            result = prune_single_section(df, sec_name, CONFIG, target_kmax=kmax)
            if result:
                all_results.append(result)

    print(f"\nSuccessfully pruned {len(all_results)} (section × kmax)")

    for kmax in CONFIG['target_kmax_list']:
        filtered = [r for r in all_results if r['target_kmax'] == kmax]
        if not filtered:
            continue
        # Summary
        summary_data = []
        for res in filtered:
            summary_data.append({
                'Section': res['section_name'],
                'Target_Kmax': res['target_kmax'],
                'Original_Nodes': res['n_assets'],
                'Selected_Nodes': res['selected_K'],
                'Train_Sharpe': f"{res['train_sharpe']:.4f}",
                'CV_Sharpe': f"{res['cv_sharpe']:.4f}",
                'Test_Sharpe': f"{res['test_sharpe']:.4f}",
                'Best_λ0': res['best_lam0'],
                'Best_λ2': res['best_lam2']
            })
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / f"All_Sections_Summary_kmax{kmax}.csv"
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n{kmax} mode summary saved: {summary_path}")

        # Detailed results with selected nodes
        detailed_rows = []
        for res in filtered:
            detailed_rows.append({
                'Section': res['section_name'],
                'Target_Kmax': res['target_kmax'],
                'Type': 'Section Info',
                'Node_Name': '',
                'Depth': '',
                'Weight': '',
                'Train_Sharpe': f"{res['train_sharpe']:.4f}",
                'CV_Sharpe': f"{res['cv_sharpe']:.4f}",
                'Test_Sharpe': f"{res['test_sharpe']:.4f}",
                'Best_λ0': res['best_lam0'],
                'Best_λ2': res['best_lam2'],
                'Original_Nodes': res['n_assets'],
                'Selected_Nodes': res['selected_K']
            })
            for node in res['selected_nodes']:
                detailed_rows.append({
                    'Section': res['section_name'],
                    'Target_Kmax': res['target_kmax'],
                    'Type': 'Selected Node',
                    'Node_Name': node['node_name'],
                    'Depth': node['depth'],
                    'Weight': f"{node['weight']:.6f}",
                    'Train_Sharpe': '',
                    'CV_Sharpe': '',
                    'Test_Sharpe': '',
                    'Best_λ0': '',
                    'Best_λ2': '',
                    'Original_Nodes': '',
                    'Selected_Nodes': ''
                })
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_path = output_dir / f"All_Sections_Detailed_Results_kmax{kmax}.csv"
        detailed_df.to_csv(detailed_path, index=False, encoding='utf-8-sig')
        print(f"Detailed results saved: {detailed_path}")

        # Test excess return series
        test_returns_dict = {}
        for res in filtered:
            if 'test_returns' in res and not res['test_returns'].empty:
                test_returns_dict[res['section_name']] = res['test_returns']
        if test_returns_dict:
            test_returns_df = pd.concat(test_returns_dict, axis=1)
            test_returns_path = output_dir / f"Test_Excess_Returns_kmax{kmax}.csv"
            test_returns_df.to_csv(test_returns_path, encoding='utf-8-sig')
            print(f"Test excess return series saved: {test_returns_path}")

    print("\n" + "=" * 120)
    print("All pruning tasks completed!")
    print("=" * 120)