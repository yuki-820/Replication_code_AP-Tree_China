"""
Step 1: Data Loader and Preprocessor

Replicates original notebook logic but keeps original dates.
All matching and aggregations use year-month only.
Output yearly files (2011-2025) in data/year/ with all required features.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from glob import glob
from tqdm import tqdm

# ========================= PATH CONFIGURATION =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_MONTHLY_DIR = PROJECT_ROOT / 'data' / 'raw' / 'monthly'
RAW_FINANCIAL_DIR = PROJECT_ROOT / 'data' / 'raw' / 'financial'
FACTOR_FILE = RAW_MONTHLY_DIR / '1995-2025三因子.csv'

OUTPUT_DIR = PROJECT_ROOT / 'data' / 'year'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== HELPER FUNCTIONS ====================
def read_csv_with_encoding(filepath):
    for enc in ['gbk', 'utf-8', 'utf-8-sig']:
        try:
            df = pd.read_csv(filepath, encoding=enc, low_memory=False,
                             dtype={'股票代码_Stkcd': str})
            df.columns = df.columns.str.strip()
            return df
        except:
            continue
    raise ValueError(f"Failed to read: {filepath}")

def clean_stkcd(series):
    return series.astype(str).str.strip().str.lstrip('Cc').str.zfill(6)

def to_year_month(date_series):
    """Convert date to YYYY-MM string for matching."""
    return pd.to_datetime(date_series).dt.strftime('%Y-%m')

def match_column(df, candidates):
    for col in df.columns:
        for c in candidates:
            if c.lower() in col.lower():
                return col
    return None

# ==================== PART 1: LOAD, CLEAN, SPLIT BY YEAR ====================
print("=" * 80)
print("Part 1: Load segmented CSV files, clean, split by year (keep original dates)")
print("=" * 80)

csv_files = [f for f in glob(str(RAW_MONTHLY_DIR / '*.csv'))
             if '三因子' not in Path(f).stem.lower()]
print(f"Found {len(csv_files)} segment files:")
for f in csv_files:
    print("   ", Path(f).name)

dfs = []
for file_path in csv_files:
    success = False
    for enc in ['gbk', 'utf-8', 'utf-8-sig']:
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False,
                             dtype={'股票代码_Stkcd': str})
            dfs.append(df)
            print(f"✓ Read successfully ({enc}): {Path(file_path).name}  | rows: {len(df):,}")
            success = True
            break
        except Exception as e:
            continue
    if not success:
        print(f"✗ FAILED TO READ: {Path(file_path).name} — all encodings failed")

if not dfs:
    raise ValueError("No segment files loaded. Check path and encoding.")

full_df = pd.concat(dfs, ignore_index=True)

# Convert dates and create year_month column
full_df['日期_Date'] = pd.to_datetime(full_df['日期_Date'], errors='coerce')
full_df['year_month'] = to_year_month(full_df['日期_Date'])

# Deduplicate: keep first record per stock per year-month
full_df.sort_values(['股票代码_Stkcd', '日期_Date'], inplace=True)
full_df.drop_duplicates(subset=['股票代码_Stkcd', 'year_month'], keep='first', inplace=True)

full_df['股票代码_Stkcd'] = full_df['股票代码_Stkcd'].astype(str).str.strip()

years = sorted(full_df['日期_Date'].dt.year.dropna().unique().astype(int))
for year in years:
    group = full_df[full_df['日期_Date'].dt.year == year].copy()
    
    # Improved listing status filter
    if '上市状态_Listedstate' in group.columns:
        cond = (
            (group['上市状态_Listedstate'] == 'Norm') |
            group['上市状态_Listedstate'].isna() |
            (group['上市状态_Listedstate'] == '')
        )
        group = group[cond]
    
    # Exclude financial industry (J)
    if '证监会行业门类代码_Csrciccd1' in group.columns:
        group = group[group['证监会行业门类代码_Csrciccd1'] != 'J']
    
    out_file = OUTPUT_DIR / f'{year}.csv'
    group.drop(columns=['year_month'], errors='ignore', inplace=True)
    group.to_csv(out_file, index=False, encoding='utf-8')
    print(f"Saved preliminary {year}.csv | rows: {len(group):,}")

print("\nPreliminary yearly split completed.\n")

# ==================== PART 2: MERGE THREE-FACTOR DATA ====================
print("=" * 80)
print("Part 2: Load three-factor file and merge into yearly files (by year-month)")
print("=" * 80)

if FACTOR_FILE.exists():
    df_factor = read_csv_with_encoding(FACTOR_FILE)
    factor_cols = ['日期_Date',
                   '市场溢酬因子__流通市值加权_Rmrf_tmv',
                   '市值因子__流通市值加权_Smb_tmv',
                   '账面市值比因子__流通市值加权_Hml_tmv']
    available_cols = ['日期_Date'] + [c for c in factor_cols[1:] if c in df_factor.columns]
    df_factor = df_factor[available_cols].copy()
    df_factor['日期_Date'] = pd.to_datetime(df_factor['日期_Date'], errors='coerce')
    df_factor['year_month'] = to_year_month(df_factor['日期_Date'])
    df_factor.drop_duplicates(subset=['year_month'], inplace=True)
    df_factor.sort_values('year_month', inplace=True)
    print(f"Three-factor data has {len(df_factor)} distinct months")

    year_files = sorted(OUTPUT_DIR.glob('[1-2][0-9][0-9][0-9].csv'))
    for file_path in year_files:
        df_year = pd.read_csv(file_path, low_memory=False, dtype={'股票代码_Stkcd': str})
        df_year['日期_Date'] = pd.to_datetime(df_year['日期_Date'], errors='coerce')
        df_year['year_month'] = to_year_month(df_year['日期_Date'])
        merged = df_year.merge(df_factor[['year_month'] + factor_cols[1:]], on='year_month', how='left')
        merged.drop(columns=['year_month'], inplace=True)
        merged.to_csv(file_path, index=False, encoding='utf-8')
        print(f"Overwrote {file_path.name} with factor data")
else:
    print("Three-factor file not found. Skipping factor merge.")

print("\nFactor merge completed.\n")

# ==================== PART 3: COMPUTE ALL FACTORS (original logic, no modification) ====================
print("=" * 80)
print("Part 3: Compute LME, Lturnover, ST_Rev, r12_2, LT_Rev (original method, no modification)")
print("=" * 80)

# Read all yearly files into a single DataFrame
year_files = sorted(OUTPUT_DIR.glob('[1-2][0-9][0-9][0-9].csv'))
all_dfs = []
for fp in year_files:
    df_y = pd.read_csv(fp, low_memory=False, dtype={'股票代码_Stkcd': str})
    df_y['日期_Date'] = pd.to_datetime(df_y['日期_Date'], errors='coerce')
    all_dfs.append(df_y)
full = pd.concat(all_dfs, ignore_index=True)
print(f"Combined sample: {len(full):,} rows")

# ==================== Core function (exactly as original script, stock-by-stock, row-by-row) ====================
def compute_all_factors(df):

    df = df.copy()
    df.sort_values(['股票代码_Stkcd', '日期_Date'], inplace=True)

    # =========================
    # 1. LME (Size)
    # =========================
    df['ME'] = df['收盘价(元)_ClPr'] * df['总股数(股)_Fullshr']
    df['LME'] = np.log(df.groupby('股票代码_Stkcd')['ME'].shift(1))

    # =========================
    # 2. Lturnover
    # =========================
    turnover_col = '总股数月换手率(%)_MonFulTurnR'
    if turnover_col in df.columns:
        df['Lturnover'] = df.groupby('股票代码_Stkcd')[turnover_col].shift(1) / 100.0
    else:
        df['Lturnover'] = np.nan

    # =========================
    # 3. Momentum & Reversal
    # =========================
    price_col = '复权价1(元)_AdjClpr1'
    if price_col not in df.columns:
        price_col = '收盘价(元)_ClPr'
        print("Warning: using unadjusted price for momentum calculation.")

    # Single-period return
    df['ret'] = df.groupby('股票代码_Stkcd')[price_col].pct_change()

    # ===== ST_Rev (t-1) - Short-term reversal
    df['ST_Rev'] = df.groupby('股票代码_Stkcd')['ret'].shift(1)

    # ===== r12_2 (t-12 ~ t-2) - Momentum factor
    # Compute r12_2 per stock: cumulative return from t-12 to t-2
    def compute_r12_2(group_df):
        group_df = group_df.sort_values('日期_Date').reset_index(drop=True)
        
        n = len(group_df)
        r12_2_values = [np.nan] * n
        
        # For each date compute r12_2 (cumulative return from t-12 to t-2)
        for i in range(n):
            # Current index corresponds to time "t"
            # We need returns from t-12 to t-2 -> indices i-12 to i-2
            start_idx = i - 12   # t-12
            end_idx = i - 2      # t-2 (inclusive)
            
            if start_idx >= 0 and end_idx >= 0 and start_idx <= end_idx:
                # Extract returns from t-12 to t-2 (11 months)
                period_rets = group_df.iloc[start_idx:end_idx+1]['ret']
                
                # Check that all returns are valid (non-NaN) and cover exactly 11 months
                if not period_rets.isna().any() and len(period_rets) == 11:
                    # Compute cumulative return
                    cumulative_ret = (1 + period_rets).prod() - 1
                    r12_2_values[i] = cumulative_ret
        
        group_df['r12_2'] = r12_2_values
        return group_df

    df = df.groupby('股票代码_Stkcd', group_keys=False).apply(compute_r12_2)

    # ===== LT_Rev (t-60 ~ t-13) - Long-term reversal
    def compute_LT_Rev(group_df):
        group_df = group_df.sort_values('日期_Date').reset_index(drop=True)
        
        n = len(group_df)
        lt_rev_values = [np.nan] * n
        
        for i in range(n):
            # Current index corresponds to time t; compute returns from t-60 to t-13
            start_idx = i - 60  # t-60
            end_idx = i - 13    # t-13 (inclusive)
            
            if start_idx >= 0 and end_idx >= 0 and start_idx <= end_idx:
                period_rets = group_df.iloc[start_idx:end_idx+1]['ret']
                
                if not period_rets.isna().any() and len(period_rets) == (end_idx - start_idx + 1):
                    cumulative_ret = (1 + period_rets).prod() - 1
                    lt_rev_values[i] = cumulative_ret
        
        group_df['LT_Rev'] = lt_rev_values
        return group_df

    df = df.groupby('股票代码_Stkcd', group_keys=False).apply(compute_LT_Rev)

    # =========================
    # Clean temporary columns
    # =========================
    df.drop(columns=['ME', 'ret'], inplace=True, errors='ignore')

    return df


# ===== Compute all factors at once =====
full = compute_all_factors(full)

# ===== Split by year and write back =====
years = sorted(full['日期_Date'].dt.year.dropna().unique().astype(int))

for y in years:
    subset = full[full['日期_Date'].dt.year == y].copy()
    out_path = OUTPUT_DIR / f"{y}.csv"
    subset.to_csv(out_path, index=False, encoding='utf-8')
    print(f"Written {y}.csv | rows: {len(subset):,}")

print("LME, Lturnover, ST_Rev, r12_2, LT_Rev calculation completed.\n")

# ==================== PART 4: COMPUTE IDIOVOL (FF3 12-MONTH ROLLING) ====================
print("=" * 80)
print("Part 4: Compute IdioVol (FF3 12‑month rolling residual volatility)")
print("=" * 80)

def add_idiovol(df):
    """
    Compute idiosyncratic volatility (IdioVol) using the Fama–French 3‑factor model.
    Strict 12‑month rolling window, no relaxation on missing values.
    """
    df = df.copy()
    df.sort_values(['股票代码_Stkcd', '日期_Date'], inplace=True)

    required = ['月收益率_Monret', '月无风险收益率_Monrfret',
                '市场溢酬因子__流通市值加权_Rmrf_tmv',
                '市值因子__流通市值加权_Smb_tmv',
                '账面市值比因子__流通市值加权_Hml_tmv']

    if any(c not in df.columns for c in required):
        print("Missing FF3 factors, IdioVol set to NaN.")
        df['IdioVol'] = np.nan
        return df

    # Excess return: Monret - Rf
    df['excess_ret'] = df['月收益率_Monret'] - df['月无风险收益率_Monrfret']

    result = []

    # Loop over stocks with progress bar
    for stk, g in tqdm(df.groupby('股票代码_Stkcd'), desc="IdioVol progress"):
        g = g.sort_values('日期_Date').reset_index(drop=True)
        n = len(g)
        idio = np.full(n, np.nan)

        # Skip if fewer than 12 observations
        if n < 12:
            g['IdioVol'] = idio
            result.append(g)
            continue

        # NumPy arrays for speed
        y_all = g['excess_ret'].values
        X_all = g[['市场溢酬因子__流通市值加权_Rmrf_tmv',
                   '市值因子__流通市值加权_Smb_tmv',
                   '账面市值比因子__流通市值加权_Hml_tmv']].values

        # Start from i = 12 to guarantee a full 12‑month window
        for i in range(12, n):
            y = y_all[i-12 : i]   # exactly 12 months
            X = X_all[i-12 : i]

            # If any NaN in the window, skip this month
            if np.isnan(y).any() or np.isnan(X).any():
                continue

            try:
                # OLS with intercept
                X_const = np.column_stack([np.ones(len(X)), X])
                beta, *_ = np.linalg.lstsq(X_const, y, rcond=None)
                resid = y - X_const @ beta
                # Standard deviation of residuals (ddof=1)
                idio[i] = np.std(resid, ddof=1)
            except:
                continue

        g['IdioVol'] = idio
        result.append(g)

    # Concatenate results back to one DataFrame
    df = pd.concat(result, ignore_index=True)
    # Lag IdioVol by one month (already based on t-12..t-1, shift for alignment)
    df['IdioVol'] = df.groupby('股票代码_Stkcd')['IdioVol'].shift(1)
    # Drop auxiliary column
    df.drop(columns=['excess_ret'], inplace=True, errors='ignore')

    return df

# ---- Reload full sample (already has LME, momentum, etc.) ----
all_dfs = []
for fp in sorted(OUTPUT_DIR.glob('[1-2][0-9][0-9][0-9].csv')):
    df_y = pd.read_csv(fp, low_memory=False, dtype={'股票代码_Stkcd': str})
    df_y['日期_Date'] = pd.to_datetime(df_y['日期_Date'], errors='coerce')
    all_dfs.append(df_y)
full = pd.concat(all_dfs, ignore_index=True)

# ---- Compute IdioVol ----
full = add_idiovol(full)

# ---- Write back by year ----
years = sorted(full['日期_Date'].dt.year.dropna().unique().astype(int))
for y in tqdm(years, desc="Writing back IdioVol"):
    subset = full[full['日期_Date'].dt.year == y].copy()
    out_path = OUTPUT_DIR / f"{y}.csv"
    subset.to_csv(out_path, index=False, encoding='utf-8')

print("IdioVol calculation completed.\n")

# ==================== PART 5: PROCESS FINANCIAL STATEMENTS AND MERGE ====================
print("=" * 80)
print("Part 5: Process balance sheet and income statement")
print("=" * 80)

def standardize_bs(df):
    col_map = {
        'ComCd': ['ComCd', '公司代码', '证券代码'],
        'EndDt': ['EndDt', '截止日期', '报告期'],
        'CashEqv': ['CashEqv', '货币资金'],
        'TotCurrAss': ['TotCurrAss', '流动资产'],
        'TotAss': ['TotAss', '总资产'],
        'ShortLoan': ['ShortLoan', '短期借款'],
        'TaxPay': ['TaxPay', '应交税费'],
        'TotCurLia': ['TotCurLia', '流动负债'],
        'StkEquPreShare': ['StkEquPreShare', '归母权益'],
        'TotShareEquit': ['TotShareEquit', '股东权益']
    }
    out = pd.DataFrame()
    for std, cand in col_map.items():
        col = match_column(df, cand)
        out[std] = df[col] if col else np.nan
    return out

def standardize_is(df):
    col_map = {
        'ComCd': ['ComCd', '公司代码'],
        'EndDt': ['EndDt', '截止日期'],
        'OpProf': ['OpProf', '营业利润']
    }
    out = pd.DataFrame()
    for std, cand in col_map.items():
        col = match_column(df, cand)
        out[std] = df[col] if col else np.nan
    return out

def standardize_keys(df, code_col, date_col):
    def clean_code(x):
        x = str(x).strip()
        if x.startswith(('C', 'c')):
            x = x[1:]
        return x.zfill(6)
    df[code_col] = df[code_col].apply(clean_code)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.date
    return df

bs_path = RAW_FINANCIAL_DIR
bs_files = [f for f in os.listdir(bs_path) if "资产负债" in f]
is_files = [f for f in os.listdir(bs_path) if "利润" in f]

if not bs_files or not is_files:
    print("Warning: Balance sheet or income statement files not found. Skipping financial ratios.")
    fin = pd.DataFrame()
else:
    bs_list = [standardize_bs(read_csv_with_encoding(os.path.join(bs_path, f))) for f in bs_files]
    is_list = [standardize_is(read_csv_with_encoding(os.path.join(bs_path, f))) for f in is_files]
    bs = pd.concat(bs_list, ignore_index=True)
    is_df = pd.concat(is_list, ignore_index=True)
    bs = standardize_keys(bs, 'ComCd', 'EndDt')
    is_df = standardize_keys(is_df, 'ComCd', 'EndDt')
    bs = bs.sort_values(['ComCd', 'EndDt']).drop_duplicates(['ComCd', 'EndDt'], keep='last')
    is_df = is_df.sort_values(['ComCd', 'EndDt']).drop_duplicates(['ComCd', 'EndDt'], keep='last')
    fin = pd.merge(bs, is_df, on=['ComCd', 'EndDt'], how='outer')
    fin = fin.sort_values(['ComCd', 'EndDt']).drop_duplicates(['ComCd', 'EndDt'], keep='last')
    fin['EndDt'] = pd.to_datetime(fin['EndDt'], errors='coerce')
    fin['ReportYear'] = fin['EndDt'].dt.year
    fin = fin.sort_values(['ComCd', 'EndDt']).drop_duplicates(['ComCd', 'ReportYear'], keep='last')
    # Correct OWC calculation
    fin['owc'] = (fin['TotCurrAss'] - fin['CashEqv'] - fin['TotCurLia']
                  + fin['ShortLoan'].fillna(0) + fin['TaxPay'].fillna(0))
    fin['StkEqu'] = fin['StkEquPreShare'].combine_first(fin['TotShareEquit'])
    fin = fin.sort_values(['ComCd', 'EndDt'])
    fin['TotAss_lag'] = fin.groupby('ComCd')['TotAss'].shift(1)
    fin['owc_lag'] = fin.groupby('ComCd')['owc'].shift(1)
    fin['Investment'] = (fin['TotAss'] / fin['TotAss_lag']) - 1
    fin['OP'] = fin['OpProf'] / fin['StkEqu']
    fin['AC'] = (fin['owc'] - fin['owc_lag']) / fin['StkEqu']
    fin = fin[['ComCd', 'ReportYear', 'TotAss', 'OpProf', 'StkEqu', 'owc',
               'Investment', 'OP', 'AC']].copy()
    print(f"Financial data ready: {len(fin)} firm-year records")

print("\n" + "=" * 80)
print("Part 6: Merge financial ratios into yearly files (by fiscal year)")
print("=" * 80)

def get_fin_year(date):
    if pd.isna(date):
        return np.nan
    y = date.year
    m = date.month
    return y - 1 if m >= 7 else y - 2

if not fin.empty:
    for year in tqdm(range(2000, 2026), desc="Financial merge progress"):
        file_path = OUTPUT_DIR / f"{year}.csv"
        if not file_path.exists():
            continue
        df = read_csv_with_encoding(file_path)
        if '日期_Date' not in df.columns:
            continue
        df['Date'] = pd.to_datetime(df['日期_Date'], errors='coerce')
        df['FinYear'] = df['Date'].apply(get_fin_year)
        if '上市公司代码_Comcd' in df.columns:
            df['上市公司代码_Comcd'] = clean_stkcd(df['上市公司代码_Comcd'])
        merged = pd.merge(
            df, fin,
            left_on=['上市公司代码_Comcd', 'FinYear'],
            right_on=['ComCd', 'ReportYear'],
            how='left'
        )
        cols_to_drop = ['ComCd', 'ReportYear', 'FinYear', 'Date']
        merged.drop(columns=[c for c in cols_to_drop if c in merged.columns], inplace=True)
        rename_map = {
            'TotAss': 'Fin_TotAss',
            'OpProf': 'Fin_OpProf',
            'StkEqu': 'Fin_StkEqu',
            'owc': 'Fin_owc',
            'Investment': 'Fin_Investment',
            'OP': 'Fin_OP',
            'AC': 'Fin_AC'
        }
        merged.rename(columns=rename_map, inplace=True)
        merged.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"  {year}.csv financial merge done")
else:
    print("No financial data to merge.")

# ==================== PART 7: COMPUTE BEME ====================
print("\n" + "=" * 80)
print("Part 7: Build Jan LME map and compute BEME")
print("=" * 80)

def safe_to_datetime(series):
    return pd.to_datetime(series, errors='coerce')

def get_date_column(df):
    if 'Date' in df.columns:
        return 'Date'
    if '日期_Date' in df.columns:
        return '日期_Date'
    for col in df.columns:
        if '日期' in col or 'Date' in col or 'date' in col.lower():
            return col
    return None

# Step 1: Build Jan LME dictionary for each (stock, year) using January records
jan_lme_dict = {}
years_range = list(range(1999, 2026))
print("Building Jan LME map...")
for year in tqdm(years_range, desc="Building map"):
    file_path = OUTPUT_DIR / f"{year}.csv"
    if not file_path.exists():
        continue
    df = read_csv_with_encoding(file_path)
    df['Date'] = safe_to_datetime(df['日期_Date'])
    df = df.dropna(subset=['Date'])
    # Keep only January rows
    df_jan = df[df['Date'].dt.month == 1].copy()
    if df_jan.empty:
        continue
    if '上市公司代码_Comcd' not in df_jan.columns:
        continue
    df_jan['ComCd'] = clean_stkcd(df_jan['上市公司代码_Comcd'])
    if 'LME' in df_jan.columns:
        df_jan['LME_num'] = pd.to_numeric(df_jan['LME'], errors='coerce')
        # For each stock, take the first LME in January (or any)
        grp = df_jan.groupby(['ComCd', df_jan['Date'].dt.year])['LME_num'].first()
        for (comcd, yr), lme in grp.items():
            jan_lme_dict[(comcd, yr)] = lme

print(f"Jan LME map built: {len(jan_lme_dict)} entries")

# Step 2: Compute BEME for each year
keep_columns = [
    '上市公司代码_Comcd', '股票代码_Stkcd', '上市状态_Listedstate',
    '证监会行业门类代码_Csrciccd1', '日期_Date', '收盘价(元)_ClPr',
    '总股数月换手率(%)_MonFulTurnR', '总股数(股)_Fullshr',
    '月收益率_Monret', '月无风险收益率_Monrfret',
    'LME', 'Lturnover', 'ST_Rev', 'r12_2', 'LT_Rev', 'IdioVol',
    'Fin_StkEqu', 'Fin_Investment', 'Fin_OP', 'Fin_AC'
]

print("\nComputing BEME...")
for year in tqdm(years_range, desc="BEME computation"):
    file_path = OUTPUT_DIR / f"{year}.csv"
    if not file_path.exists():
        continue
    df = read_csv_with_encoding(file_path)
    date_col = get_date_column(df)
    if date_col is None:
        continue
    df['Date'] = safe_to_datetime(df[date_col])
    df = df.dropna(subset=['Date']).reset_index(drop=True)
    available_keep = [col for col in keep_columns if col in df.columns]
    df = df[available_keep + ['Date']].copy()
    if year == 1999:
        df['BEME'] = np.nan
    else:
        if 'Fin_StkEqu' not in df.columns or '上市公司代码_Comcd' not in df.columns:
            df['BEME'] = np.nan
        else:
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['ComCd'] = clean_stkcd(df['上市公司代码_Comcd'])
            # BE year (fiscal year)
            df['BE_Year'] = df.apply(lambda r: r['Year'] - 1 if r['Month'] <= 6 else r['Year'], axis=1)
            # ME year for Jan LME (December of previous year)
            df['ME_Year'] = df.apply(lambda r: r['Year'] - 1 if r['Month'] <= 6 else r['Year'], axis=1)
            df['Jan_LME'] = df.apply(lambda r: jan_lme_dict.get((r['ComCd'], r['ME_Year']), np.nan), axis=1)
            df['BE'] = pd.to_numeric(df['Fin_StkEqu'], errors='coerce')
            df['BEME'] = df['BE'] / np.exp(df['Jan_LME'])
            drop_cols = ['Year', 'Month', 'BE_Year', 'ME_Year', 'ComCd', 'Jan_LME', 'Date']
            df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    # Remove temporary Date column
    df.drop(columns=['Date'], inplace=True, errors='ignore')
    # Standardize stock codes
    if '上市公司代码_Comcd' in df.columns:
        df['上市公司代码_Comcd'] = clean_stkcd(df['上市公司代码_Comcd'])
    if '股票代码_Stkcd' in df.columns:
        df['股票代码_Stkcd'] = clean_stkcd(df['股票代码_Stkcd'])
    # Overwrite file
    df.to_csv(file_path, index=False, encoding='utf-8-sig')

print("BEME computation completed.\n")

# ==================== PART 8: DELETE FILES BEFORE 2011 ====================
print("=" * 80)
print("Part 8: Delete files with year < 2011 (keep 2011 and later)")
print("=" * 80)

for file_path in OUTPUT_DIR.glob('[1-2][0-9][0-9][0-9].csv'):
    year = int(file_path.stem)
    if year < 2011:
        file_path.unlink()
        print(f"Deleted {file_path.name} (year < 2011)")
    else:
        print(f"Kept {file_path.name}")

print("\n" + "=" * 80)
print("All data processing completed successfully!")
print(f"Final files are located at {OUTPUT_DIR}, years 2011-2025.")
print("=" * 80)
