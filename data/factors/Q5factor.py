from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Parameter Configuration
# =============================================================================
INPUT_FILE = Path("q5prepare_clean.csv")
OUTPUT_DATA = Path("q5data.csv")
OUTPUT_FACTOR = Path("q5factor.csv")        # Five-factor output

START_MONTH = pd.Period("2021-01", freq="M")
END_MONTH = pd.Period("2025-12", freq="M")

RETURN_IS_DECIMAL = False           # Changed to False, output as decimal rather than percentage
ACCOUNTING_LAG_MONTHS = 4
MIN_REG_MONTHS = 30
MAX_REG_MONTHS = 120
WINSORIZE_PCT = 0.01
OPEXP_INCLUDES_ADMEXP = False
OPEXP_INCLUDES_RDEXP = False

# Field mapping
COL_MAP = {
    'id': 'ComCd',
    'date': 'Date',
    'ret': 'Monret',
    'rf': 'Monrfret',
    'clpr': 'ClPr',
    'fullshr': 'Fullshr',
    'totass': 'TotAss',
    'oprev': 'OpRev',
    'opcost': 'OpCost',
    'opexp': 'OpExp',
    'admexp': 'AdmExp',
    'rdexp': 'RDExp',
    'accrec': 'AccRec',
    'advpay': 'AdvPay',
    'inventories': 'Inventories',
    'contract_lia': 'ContractLia',
    'accpay': 'AccPay',
    'salarpay': 'SalarPay',
    'taxpay': 'TaxPay',
    'othpay_intr_div': 'OthPayIntrDivid',
    'shortloan': 'ShortLoan',
    'shortbond_fina_pay': 'ShortBondFinaPay',
    'shortbond_pay': 'ShortBondPay',
    'ncurrliab_one': 'NCurrLiabOne',
    'longloan': 'LongLoan',
    'bondpay': 'BondPay',
    'leaslia': 'LeasLia',
    'longaccpay': 'LongAccPay',
    'totshareequit_parent': 'TotShareEquitParent',
    'totshareequit': 'TotShareEquit',
    'np_parent_q': 'NPParentComp_Q',
}

# =============================================================================
# Utility Functions
# =============================================================================
def first_available(frame, columns):
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for col in columns:
        if col in frame.columns:
            x = pd.to_numeric(frame[col], errors="coerce")
            out = out.where(out.notna(), x)
    return out

def vw_return(frame, ret_col="Ret", weight_col="ME_lag"):
    x = frame[[ret_col, weight_col]].copy()
    x = x.dropna(subset=[ret_col, weight_col])
    x = x.loc[x[weight_col] > 0]
    if x.empty:
        return np.nan
    total_weight = x[weight_col].sum()
    if total_weight <= 0:
        return np.nan
    return (x[ret_col] * x[weight_col]).sum() / total_weight

def assign_30_40_30(series):
    result = pd.Series(pd.NA, index=series.index, dtype="string")
    valid = series.dropna()
    if len(valid) < 3:
        return result
    p30 = valid.quantile(0.30)
    p70 = valid.quantile(0.70)
    result.loc[series <= p30] = "L"
    result.loc[(series > p30) & (series <= p70)] = "M"
    result.loc[series > p70] = "H"
    return result

def winsorize_series(series, lower_pct=0.01, upper_pct=0.99):
    if series.dropna().empty:
        return series
    low = series.quantile(lower_pct)
    high = series.quantile(upper_pct)
    return series.clip(lower=low, upper=high)

# =============================================================================
# 1. Read raw data (keep full history)
# =============================================================================
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", low_memory=False)

needed = set(COL_MAP.values()) - {'Monret', 'Monrfret', 'ClPr', 'Fullshr'}
needed.update(['ComCd', 'A_StkCd', 'Date', 'TotAss', 'NPParentComp_Q', 'TotShareEquitParent'])
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df[COL_MAP['id']] = df[COL_MAP['id']].astype("string").str.strip().str.zfill(6)
df["A_StkCd"] = df["A_StkCd"].astype("string").str.strip()
df[COL_MAP['date']] = pd.to_datetime(df[COL_MAP['date']], errors="coerce")
df = df.dropna(subset=[COL_MAP['id'], "A_StkCd", COL_MAP['date']]).copy()
df["Month"] = df[COL_MAP['date']].dt.to_period("M")

ret_raw = first_available(df, [COL_MAP['ret'], 'Ret', 'Monret'])
rf_raw = first_available(df, [COL_MAP['rf'], 'RF', 'Monrfret'])
if RETURN_IS_DECIMAL:
    df["Ret"] = ret_raw * 100
    df["RF"] = rf_raw * 100
else:
    df["Ret"] = ret_raw
    df["RF"] = rf_raw

df["ME"] = first_available(df, ["ME"])
if {COL_MAP['clpr'], COL_MAP['fullshr']}.issubset(df.columns):
    price = pd.to_numeric(df[COL_MAP['clpr']], errors="coerce").abs()
    shares = pd.to_numeric(df[COL_MAP['fullshr']], errors="coerce")
    me_fallback = price * shares
    df["ME"] = df["ME"].where(df["ME"].gt(0), me_fallback)

fin_cols = [v for k, v in COL_MAP.items() if k not in ['id', 'date', 'ret', 'rf', 'clpr', 'fullshr']]
for col in fin_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values(["A_StkCd", "Month", COL_MAP['date']], kind="stable")
df = df.drop_duplicates(["A_StkCd", "Month"], keep="last").copy()

# =============================================================================
# 2. Compute I/A and ROE
# =============================================================================
company_monthly = df.groupby([COL_MAP['id'], "Month"], sort=False).agg(
    TotAss=("TotAss", "last"),
    NPParentComp_Q=("NPParentComp_Q", "last"),
    TotShareEquitParent=("TotShareEquitParent", "last")
).reset_index()

company_monthly["YearMonth"] = company_monthly["Month"].astype(str)
company_monthly["YearMonth_lag"] = (company_monthly["Month"] - 12).astype(str)
ia_lookup = company_monthly[["ComCd", "YearMonth", "TotAss"]].copy()
ia_lookup = ia_lookup.rename(columns={"YearMonth": "YearMonth_lag", "TotAss": "TotAss_lag12"})
company_monthly = company_monthly.merge(ia_lookup, on=["ComCd", "YearMonth_lag"], how="left")
company_monthly["IA"] = (company_monthly["TotAss"] - company_monthly["TotAss_lag12"]) / company_monthly["TotAss_lag12"]
company_monthly.loc[(company_monthly["TotAss"].le(0)) | (company_monthly["TotAss_lag12"].le(0)), "IA"] = np.nan

company_monthly["ROE"] = company_monthly["NPParentComp_Q"] / company_monthly["TotShareEquitParent"]
company_monthly.loc[company_monthly["TotShareEquitParent"].le(0), "ROE"] = np.nan

df_out = df.merge(company_monthly[[COL_MAP['id'], "Month", "IA", "ROE"]], on=[COL_MAP['id'], "Month"], how="left")

# =============================================================================
# 3. Build firm-level monthly panel (firm)
# =============================================================================
df = df.sort_values(["A_StkCd", "Month"], kind="stable")
df["ME_lag_security"] = df.groupby("A_StkCd", sort=False)["ME"].shift(1)
df["PrevMonth"] = df.groupby("A_StkCd", sort=False)["Month"].shift(1)
df.loc[df["PrevMonth"] != df["Month"] - 1, "ME_lag_security"] = np.nan

firm_rows = []
for (comcd, month), group in df.groupby([COL_MAP['id'], "Month"], sort=True):
    firm_ret = vw_return(
        group.rename(columns={"ME_lag_security": "ME_lag"}),
        ret_col="Ret", weight_col="ME_lag"
    )
    firm_me = group["ME"].where(group["ME"] > 0).sum(min_count=1)
    firm_me_lag = group["ME_lag_security"].where(group["ME_lag_security"] > 0).sum(min_count=1)
    firm_rf = group["RF"].median()
    fin_data = {}
    for col in fin_cols:
        if col in group.columns:
            vals = group[col].dropna()
            fin_data[col] = vals.iloc[-1] if not vals.empty else np.nan
    firm_rows.append({
        "ComCd": comcd, "Month": month,
        "Ret": firm_ret, "ME": firm_me, "ME_lag": firm_me_lag, "RF": firm_rf,
        **fin_data
    })

firm = pd.DataFrame(firm_rows)
firm = firm.sort_values(["ComCd", "Month"]).reset_index(drop=True)
firm = firm.merge(company_monthly[["ComCd", "Month", "IA", "ROE"]], on=["ComCd", "Month"], how="left")

# =============================================================================
# 4. Original four-factor construction
# =============================================================================
sample = firm.loc[firm["Month"].between(START_MONTH, END_MONTH)].copy()

def assign_groups(monthly_df):
    df = monthly_df.copy()
    med = df["ME_lag"].median()
    df["SizeGroup"] = np.where(df["ME_lag"] <= med, "S", "B")
    df["IAGroup"] = assign_30_40_30(df["IA"])
    df["RoeGroup"] = assign_30_40_30(df["ROE"])
    return df

grouped = sample.groupby("Month", sort=False, group_keys=False).apply(assign_groups)
grouped = grouped.reset_index(drop=True)

factor_list = []
for month, g in grouped.groupby("Month", sort=True):
    market = g.loc[g["Ret"].notna() & g["ME_lag"].gt(0)]
    rm = vw_return(market, ret_col="Ret", weight_col="ME_lag")
    rf = market["RF"].dropna().median()
    rmrf = rm - rf if pd.notna(rm) and pd.notna(rf) else np.nan

    q = g.loc[
        g["Ret"].notna() & g["ME_lag"].gt(0) &
        g["SizeGroup"].notna() & g["IAGroup"].notna() & g["RoeGroup"].notna()
    ]
    portfolios = {}
    for size in ["S", "B"]:
        for ia in ["L", "M", "H"]:
            for roe in ["L", "M", "H"]:
                sub = q[(q["SizeGroup"]==size) & (q["IAGroup"]==ia) & (q["RoeGroup"]==roe)]
                portfolios[(size, ia, roe)] = vw_return(sub, "Ret", "ME_lag")

    small_ret = [portfolios[("S", ia, roe)] for ia in ["L","M","H"] for roe in ["L","M","H"]]
    big_ret   = [portfolios[("B", ia, roe)] for ia in ["L","M","H"] for roe in ["L","M","H"]]
    r_me = np.mean(small_ret) - np.mean(big_ret) if all(np.isfinite(small_ret + big_ret)) else np.nan

    low_ia  = [portfolios[(s, "L", roe)] for s in ["S","B"] for roe in ["L","M","H"]]
    high_ia = [portfolios[(s, "H", roe)] for s in ["S","B"] for roe in ["L","M","H"]]
    r_ia = np.mean(low_ia) - np.mean(high_ia) if all(np.isfinite(low_ia + high_ia)) else np.nan

    high_roe = [portfolios[(s, ia, "H")] for s in ["S","B"] for ia in ["L","M","H"]]
    low_roe  = [portfolios[(s, ia, "L")] for s in ["S","B"] for ia in ["L","M","H"]]
    r_roe = np.mean(high_roe) - np.mean(low_roe) if all(np.isfinite(high_roe + low_roe)) else np.nan

    factor_list.append({
        "Date": month.to_timestamp("M"),
        "Rm": rm, "RF": rf, "RmRf": rmrf,
        "R_ME": r_me, "R_IA": r_ia, "R_ROE": r_roe,
        "N_Market": len(market), "N_QFactor": len(q)
    })

factors = pd.DataFrame(factor_list).sort_values("Date").reset_index(drop=True)

# =============================================================================
# 5. EG factor construction
# =============================================================================
print("\n===== Starting EG factor construction =====")

fin_vars = [COL_MAP[c] for c in ['oprev','opcost','opexp','admexp','rdexp',
                                  'accrec','advpay','inventories',
                                  'contract_lia','accpay','salarpay','taxpay',
                                  'othpay_intr_div',
                                  'shortloan','shortbond_fina_pay','shortbond_pay',
                                  'ncurrliab_one','longloan','bondpay','leaslia',
                                  'longaccpay','totass','totshareequit_parent',
                                  'totshareequit']]
fin_vars = [v for v in fin_vars if v in firm.columns]

firm = firm.sort_values(["ComCd", "Month"])
for col in fin_vars:
    firm[col + "_lag"] = firm.groupby("ComCd")[col].shift(ACCOUNTING_LAG_MONTHS)

bs_vars = ['AccRec','AdvPay','Inventories','ContractLia','AccPay',
           'SalarPay','TaxPay','OthPayIntrDivid']
for col in bs_vars:
    col_raw = COL_MAP.get(col.lower(), None)
    if col_raw and col_raw in firm.columns:
        lag_col = col_raw + "_lag"
        firm[col + "_lag12"] = firm.groupby("ComCd")[col_raw].shift(ACCOUNTING_LAG_MONTHS + 12)
        firm["d_" + col] = firm[lag_col] - firm[col + "_lag12"]

debt_components = [COL_MAP[k] for k in ['shortloan','shortbond_fina_pay','shortbond_pay',
                                        'ncurrliab_one','longloan','bondpay','leaslia','longaccpay']
                   if COL_MAP[k] in firm.columns]
firm["Debt"] = firm[debt_components].sum(axis=1, min_count=1)
all_missing = firm[debt_components].isna().all(axis=1)
firm.loc[all_missing, "Debt"] = np.nan

firm["ME_lag1"] = firm.groupby("ComCd")["ME"].shift(1)
firm["q_proxy"] = (firm["ME_lag1"] + firm["Debt"]) / firm["TotAss_lag"]
firm.loc[firm["TotAss_lag"].le(0), "q_proxy"] = np.nan

# Cop: treat missing components as 0
Cop_num = (firm[COL_MAP['oprev'] + "_lag"].fillna(0)
           - firm[COL_MAP['opcost'] + "_lag"].fillna(0)
           - firm[COL_MAP['opexp'] + "_lag"].fillna(0))
if not OPEXP_INCLUDES_ADMEXP:
    Cop_num -= firm[COL_MAP['admexp'] + "_lag"].fillna(0)
if not OPEXP_INCLUDES_RDEXP:
    Cop_num += firm[COL_MAP['rdexp'] + "_lag"].fillna(0)

for item, sign in [('d_AccRec', -1), ('d_Inventories', -1), ('d_AdvPay', -1),
                   ('d_ContractLia', 1), ('d_AccPay', 1), ('d_SalarPay', 1),
                   ('d_TaxPay', 1), ('d_OthPayIntrDivid', 1)]:
    if item in firm.columns:
        Cop_num += sign * firm[item].fillna(0)

firm["TotAss_lag5"] = firm.groupby("ComCd")["TotAss"].shift(5)
firm["Cop"] = Cop_num / firm["TotAss_lag5"]
firm.loc[firm["TotAss_lag5"].le(0), "Cop"] = np.nan

firm["ROE_lag"] = firm.groupby("ComCd")["ROE"].shift(ACCOUNTING_LAG_MONTHS)
firm["ROE_lag12"] = firm.groupby("ComCd")["ROE"].shift(ACCOUNTING_LAG_MONTHS + 12)
firm["dROE"] = firm["ROE_lag"] - firm["ROE_lag12"]
firm["dROE_missing_flag"] = firm["ROE_lag"].isna() | firm["ROE_lag12"].isna()

# d1_IA based on true date offset
firm['Month_target'] = firm['Month'] + 12
ia_lead = firm[['ComCd', 'Month', 'IA']].rename(columns={'Month': 'Month_target', 'IA': 'IA_lead12'})
firm = firm.merge(ia_lead, on=['ComCd', 'Month_target'], how='left')
firm['d1_IA'] = firm['IA_lead12'] - firm['IA']
firm.drop(columns=['Month_target'], inplace=True)

# Cross-sectional regression
reg_data = firm[["ComCd", "Month", "q_proxy", "Cop", "dROE", "d1_IA"]].dropna().copy()
reg_data = reg_data.sort_values(["Month", "ComCd"])

coef_list = []
for month, grp in reg_data.groupby("Month"):
    n = len(grp)
    if n < 10:
        continue
    for col in ["q_proxy", "Cop", "dROE"]:
        grp[col] = winsorize_series(grp[col], WINSORIZE_PCT, 1-WINSORIZE_PCT)
    X = grp[["q_proxy", "Cop", "dROE"]].values
    y = grp["d1_IA"].values
    try:
        model = LinearRegression().fit(X, y)
        coef_list.append({
            "Month": month,
            "alpha": model.intercept_,
            "beta_q": model.coef_[0],
            "beta_Cop": model.coef_[1],
            "beta_dROE": model.coef_[2],
            "N": n
        })
    except:
        continue

coef_df = pd.DataFrame(coef_list).sort_values("Month")
print(f"Valid regression months: {len(coef_df)}, earliest: {coef_df['Month'].min()}, latest: {coef_df['Month'].max()}")

# Calculate EG
firm = firm.sort_values(["ComCd", "Month"])
for t_month in firm["Month"].unique():
    hist_coefs = coef_df[coef_df["Month"] < t_month].tail(MAX_REG_MONTHS)
    if len(hist_coefs) < MIN_REG_MONTHS:
        firm.loc[firm["Month"] == t_month, "EG"] = np.nan
        firm.loc[firm["Month"] == t_month, "EG_n_reg_months"] = len(hist_coefs)
        continue
    avg_alpha = hist_coefs["alpha"].mean()
    avg_beta_q = hist_coefs["beta_q"].mean()
    avg_beta_Cop = hist_coefs["beta_Cop"].mean()
    avg_beta_dROE = hist_coefs["beta_dROE"].mean()
    n_reg = len(hist_coefs)

    mask = firm["Month"] == t_month
    tmp = firm.loc[mask, ["q_proxy", "Cop", "dROE"]].copy()
    for col in ["q_proxy", "Cop", "dROE"]:
        tmp[col] = winsorize_series(tmp[col], WINSORIZE_PCT, 1-WINSORIZE_PCT)
    firm.loc[mask, "EG"] = (avg_alpha +
                            avg_beta_q * tmp["q_proxy"] +
                            avg_beta_Cop * tmp["Cop"] +
                            avg_beta_dROE * tmp["dROE"])
    firm.loc[mask, "EG_n_reg_months"] = n_reg
    firm.loc[mask, "EG_alpha"] = avg_alpha
    firm.loc[mask, "EG_beta_q"] = avg_beta_q
    firm.loc[mask, "EG_beta_Cop"] = avg_beta_Cop
    firm.loc[mask, "EG_beta_dROE"] = avg_beta_dROE

# EG factor portfolios
eg_sample = firm.loc[firm["Month"].between(START_MONTH, END_MONTH)].copy()
eg_sample = eg_sample[eg_sample["EG"].notna() & eg_sample["ME_lag1"].gt(0) & eg_sample["Ret"].notna()]

eg_factor_list = []
for month, grp in eg_sample.groupby("Month"):
    med_me = grp["ME_lag1"].median()
    grp["Size_EG"] = np.where(grp["ME_lag1"] <= med_me, "S", "B")
    grp["EG_group"] = assign_30_40_30(grp["EG"])
    combos = {}
    for size in ["S", "B"]:
        for eg_g in ["L", "M", "H"]:
            sub = grp[(grp["Size_EG"]==size) & (grp["EG_group"]==eg_g)]
            ret = vw_return(sub, ret_col="Ret", weight_col="ME_lag1")
            combos[(size, eg_g)] = ret
    low_ret = [combos[("S","L")], combos[("B","L")]]
    high_ret = [combos[("S","H")], combos[("B","H")]]
    if all(np.isfinite(low_ret + high_ret)):
        eg_factor = 0.5 * (high_ret[0] + high_ret[1]) - 0.5 * (low_ret[0] + low_ret[1])
        valid = 1
    else:
        eg_factor = np.nan
        valid = 0
    rf_median = grp["RF"].median()
    eg_factor_list.append({
        "Date": month.to_timestamp("M"),
        "EG_factor": eg_factor,
        "EG_factor_excess": eg_factor - rf_median if pd.notna(eg_factor) else np.nan,
        "Monrfret": rf_median,
        "factor_valid_flag": valid
    })

df_eg_factor = pd.DataFrame(eg_factor_list).sort_values("Date")

# =============================================================================
# 6. Output final files
# =============================================================================
# q5data.csv
eg_merge = firm[["ComCd", "Month", "EG"]].drop_duplicates()
df_final = df_out.merge(eg_merge, on=["ComCd", "Month"], how="left")
df_final.to_csv(OUTPUT_DATA, index=False, encoding="utf-8-sig", float_format="%.8f")
print(f"Output q5data.csv: {OUTPUT_DATA.resolve()}, rows: {len(df_final)}")

# Five-factor file: keep only Date, RmRf, R_ME, R_IA, R_ROE, EG_factor
factors = factors.merge(df_eg_factor[["Date", "EG_factor"]], on="Date", how="left")
factor_out_cols = ["Date", "RmRf", "R_ME", "R_IA", "R_ROE", "EG_factor"]
factors[factor_out_cols].to_csv(OUTPUT_FACTOR, index=False, encoding="utf-8-sig", float_format="%.8f")
print(f"Output five-factor file: {OUTPUT_FACTOR.resolve()}, rows: {len(factors)}")

print("\nFactor missing counts:")
print(factors[factor_out_cols[1:]].isna().sum())
print("\nEarliest available month for EG_factor:", factors.loc[factors["EG_factor"].notna(), "Date"].min())