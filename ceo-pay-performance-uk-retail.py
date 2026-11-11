import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.graphics.gofplots import qqplot

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

# ============================== CONFIG ===================================
FILE_PATH = r"C:\Users\brian\Downloads\New folder (5)\WRDS DATA.xlsx"
COMPENSATION_SHEET = "Sheet1"          # board / director remuneration extract
FINANCIAL_SHEET = "Sheet2"             # firm-year financial panel

# Which Sheet1 column to treat as "CEO total remuneration" for the pay
# variable. Alternatives available in the file: "totaldirectcomp" (cash-only:
# salary+bonus+other, no pension/equity) or "totremperiod" (total
# remuneration for the reporting period, may include equity-at-risk).
# "totalcompensation" is used by default as the closest analogue to the
# original "Total Remuneration/Total Compensation" field.
CEO_PAY_COLUMN = "totalcompensation"

# Expected reporting currency for CEO pay. Rows in a different currency are
# NOT converted (no FX series is supplied) -- they are flagged in the
# [data check] output so you can decide whether to convert or exclude them.
CEO_PAY_CURRENCY = "GBP"

# Regex used to identify a CEO row from the `rolename` column (case-
# insensitive). Broadened slightly to also catch "Group Chief Executive" /
# "Chief Executive Officer" / "CEO" while trying to avoid "Deputy CEO" or
# "Vice Chairman"-type false positives; adjust if your rolename values look
# different from the sample rows.
CEO_ROLE_PATTERN = r"\bchief executive\b|\bceo\b"
EXCLUDE_ROLE_PATTERN = r"\bdeputy\b|\bformer\b"

WINSOR_LIMITS = (0.01, 0.01)          # 1st / 99th percentile, per Section 3.4
EXCLUDE_YEARS = [2020, 2021]           # COVID-19 exclusion robustness check
RANDOM_SEED = 42
N_BOOTSTRAP = 4999                      # wild-cluster bootstrap replications
SHOW_PLOTS = True                       # set False to skip all plotting
PLOT_DIAGNOSTICS_FOR = "Baseline (full sample)"  # which run_all_models() label
                                                  # gets per-model diagnostic plots

# Key modelling variables whose missing values get filled with 0 (rather
# than dropped) once the panel is built -- see construct_variables().
FILL_ZERO_COLS = ["ROA", "ROE", "CEOPay", "LaggedCEOPay", "FirmSize", "Leverage"]
# ===========================================================================

np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# 0. NAME NORMALISATION (shared by both loaders, used as the merge key)
# ---------------------------------------------------------------------------

_LEGAL_SUFFIXES = re.compile(r"\b(PLC|LTD|LIMITED|GROUP)\b\.?", re.IGNORECASE)


def _normalise_company_key(name) -> str:
    """Build a robust, case/spacing/legal-suffix-insensitive merge key.
    Also special-cases Card Factory's Executive/Non-Executive board listings
    to merge into a single company, as instructed for the original data."""
    if pd.isna(name):
        return name
    s = str(name).strip().upper()
    if s.startswith("CARD FACTORY"):
        return "CARD FACTORY"
    s = _LEGAL_SUFFIXES.sub("", s)
    s = re.sub(r"[^A-Z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# 1. DATA LOADING AND CLEANING  (Sections 3.2 - 3.4)
# ---------------------------------------------------------------------------

def load_financial_data() -> pd.DataFrame:
    df = pd.read_excel(FILE_PATH, sheet_name=FINANCIAL_SHEET)
    df.columns = [c.strip() for c in df.columns]
    df["Company"] = df["Company"].astype(str).str.strip()
    df["CompanyKey"] = df["Company"].apply(_normalise_company_key)
    df["Year"] = df["Year"].astype(int)

    numeric_cols = ["Profit After Tax", "Total Assets", "Equity", "Total Debt",
                     "Revenue", "Operating Profit", "ROA", "ROE", "Firm Size"]
    for c in numeric_cols:
        if c not in df.columns:
            continue
        before_na = df[c].isna().sum()
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )
        newly_bad = df[c].isna().sum() - before_na
        if newly_bad > 0:
            print(f"[data check] '{c}' (Sheet2): {newly_bad} non-numeric value(s) "
                  f"coerced to NaN.")

    dupes = df.duplicated(subset=["CompanyKey", "Year"]).sum()
    if dupes:
        print(f"[data check] WARNING: Sheet2 has {dupes} duplicate Company-Year rows "
              f"after name normalisation -- check for near-duplicate company names.")

    return df


def identify_ceo_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Select individual (non-aggregate) rows that represent the CEO role.
    Prints diagnostics so the classification can be checked against the raw
    file, since Sheet1 mixes named-director rows with board-level aggregate
    rows (e.g. 'ED Average', 'NED Average') that must NOT be treated as a
    CEO observation."""
    print(f"\n{'-' * 90}\n[data check] Unique 'rowtype' values in {COMPENSATION_SHEET} "
          f"(top 20 by frequency)\n{'-' * 90}")
    print(df["rowtype"].value_counts().head(20))

    print(f"\n{'-' * 90}\n[data check] Unique 'rolename' values in {COMPENSATION_SHEET} "
          f"(top 30 by frequency)\n{'-' * 90}")
    print(df["rolename"].value_counts().head(30))

    is_named = (
        df["directorname"].notna()
        & (df["directorname"].astype(str).str.strip().str.lower() != "n.a.")
    )
    role_text = df["rolename"].astype(str)
    is_ceo_role = role_text.str.contains(CEO_ROLE_PATTERN, case=False, regex=True, na=False)
    is_excluded = role_text.str.contains(EXCLUDE_ROLE_PATTERN, case=False, regex=True, na=False)

    ceo_mask = is_named & is_ceo_role & ~is_excluded
    ceo_df = df.loc[ceo_mask].copy()

    print(f"\n[data check] {ceo_mask.sum()} of {len(df)} rows in {COMPENSATION_SHEET} "
          f"classified as individual CEO rows (named director + role matching "
          f"'{CEO_ROLE_PATTERN}').")
    if ceo_mask.sum() == 0:
        print("[data check] WARNING: no CEO rows identified. Inspect the 'rolename' "
              "listing above and adjust CEO_ROLE_PATTERN in CONFIG.")

    return ceo_df


def load_compensation_data() -> pd.DataFrame:
    """Build one CEO-compensation observation per company-year from Sheet1.
    Where more than one CEO served during a financial year, remuneration
    attributable to the CEO role is aggregated (Section 3.2)."""
    df = pd.read_excel(FILE_PATH, sheet_name=COMPENSATION_SHEET)
    df.columns = [c.strip() for c in df.columns]

    df["boardname"] = df["boardname"].astype(str).str.strip()
    df["CompanyKey"] = df["boardname"].apply(_normalise_company_key)

    df["annualreportdate"] = pd.to_datetime(
        df["annualreportdate"], dayfirst=True, errors="coerce"
    )
    n_bad_dates = df["annualreportdate"].isna().sum()
    if n_bad_dates:
        print(f"[data check] {n_bad_dates} row(s) in {COMPENSATION_SHEET} had an "
              f"unparseable 'annualreportdate' and will be dropped.")
    df = df.dropna(subset=["annualreportdate"])
    df["Year"] = df["annualreportdate"].dt.year.astype(int)

    ceo_df = identify_ceo_rows(df)

    # Currency check -- flag, do not silently convert.
    if "currency" in ceo_df.columns:
        bad_ccy = ceo_df.loc[ceo_df["currency"].astype(str).str.upper() != CEO_PAY_CURRENCY]
        if len(bad_ccy):
            ccy_counts = bad_ccy["currency"].value_counts()
            print(f"\n[data check] WARNING: {len(bad_ccy)} CEO row(s) reported in a "
                  f"currency other than {CEO_PAY_CURRENCY} (no FX conversion applied):")
            print(ccy_counts)
            print("[data check] Affected firm-years:")
            print(bad_ccy[["boardname", "Year", "currency"]].drop_duplicates()
                  .to_string(index=False))

    comp_components = [
        "salary", "bonus", "other", "penempcon", CEO_PAY_COLUMN,
        "toteqatrisk", "totaldirectcomp", "totremperiod",
    ]
    comp_components = [c for c in comp_components if c in ceo_df.columns]

    # Force numeric dtype; stray text (e.g. 'n/a', '-', thousands separators
    # stored as text) would otherwise make the column object-dtype and crash
    # groupby().sum() with "TypeError: can only concatenate str (not int) to str".
    for c in comp_components:
        before_na = ceo_df[c].isna().sum()
        ceo_df[c] = pd.to_numeric(
            ceo_df[c].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )
        after_na = ceo_df[c].isna().sum()
        newly_bad = after_na - before_na
        if newly_bad > 0:
            print(f"[data check] '{c}' ({COMPENSATION_SHEET}): {newly_bad} non-numeric "
                  f"value(s) coerced to NaN.")

    agg = (
        ceo_df.groupby(["CompanyKey", "Year"], as_index=False)[comp_components]
        .sum(min_count=1)
    )
    n_ceos = (
        ceo_df.groupby(["CompanyKey", "Year"])
        .size()
        .rename("n_ceo_rows")
        .reset_index()
    )
    agg = agg.merge(n_ceos, on=["CompanyKey", "Year"], how="left")

    # Keep one representative board name per key purely for readable diagnostics
    board_names = ceo_df.groupby("CompanyKey")["boardname"].first().rename("BoardName")
    agg = agg.merge(board_names, on="CompanyKey", how="left")

    agg = agg.rename(columns={CEO_PAY_COLUMN: "TotalRemuneration"})
    return agg


def build_panel() -> pd.DataFrame:
    fin = load_financial_data()
    comp = load_compensation_data()

    panel = fin.merge(comp, on=["CompanyKey", "Year"], how="inner", validate="one_to_one")

    dropped_fin = set(zip(fin["CompanyKey"], fin["Year"])) - set(zip(panel["CompanyKey"], panel["Year"]))
    if dropped_fin:
        print(f"\n[data check] {len(dropped_fin)} firm-year(s) in {FINANCIAL_SHEET} had no "
              f"matching CEO compensation record in {COMPENSATION_SHEET} and were dropped:")
        print(sorted(dropped_fin))

    dropped_comp = set(zip(comp["CompanyKey"], comp["Year"])) - set(zip(panel["CompanyKey"], panel["Year"]))
    if dropped_comp:
        print(f"\n[data check] {len(dropped_comp)} CEO firm-year(s) in {COMPENSATION_SHEET} had "
              f"no matching row in {FINANCIAL_SHEET} (check company-name spelling) and were dropped:")
        print(sorted(dropped_comp))

    dupes = panel.duplicated(subset=["CompanyKey", "Year"]).sum()
    if dupes:
        print(f"[data check] WARNING: {dupes} duplicate firm-year rows found in the merged panel.")

    return panel


def construct_variables(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df = df.sort_values(["CompanyKey", "Year"]).reset_index(drop=True)

    # Recomputed performance and independent variables (Section 3.3), from
    # the raw Sheet2 accounting components -- kept identical to the original
    # methodology rather than trusting Sheet2's pre-computed columns blindly.
    df["ROA"] = df["Profit After Tax"] / df["Total Assets"]
    df["ROE"] = df["Profit After Tax"] / df["Equity"]
    df["FirmSize"] = np.log(df["Total Assets"])
    df["Leverage"] = df["Total Debt"] / df["Total Assets"]

    # Cross-check against Sheet2's own pre-computed ROA / ROE / Firm Size,
    # where present, and flag any material discrepancy.
    for recomputed, provided, tol in [("ROA", "ROA", 1e-3), ("ROE", "ROE", 1e-3),
                                       ("FirmSize", "Firm Size", 1e-3)]:
        if provided in df.columns:
            diff = (df[recomputed] - df[provided]).abs()
            n_off = int((diff > tol).sum())
            if n_off:
                print(f"[data check] {n_off} firm-year(s) where recomputed {recomputed} differs "
                      f"from Sheet2's pre-computed '{provided}' by more than {tol}.")

    # CEO pay must be positive to take logs; flag and null non-positive pay
    # (a log of a non-positive number is undefined, so it cannot itself be
    # set to 0 -- it is left as NaN here and then swept up by the general
    # zero-fill step below, same as any other missing value).
    non_positive = df["TotalRemuneration"] <= 0
    if non_positive.any():
        print(f"[data check] {non_positive.sum()} firm-year(s) have non-positive "
              f"TotalRemuneration; CEOPay will be filled with 0 for these rows.")
    df.loc[non_positive, "TotalRemuneration"] = np.nan

    df["CEOPay"] = np.log(df["TotalRemuneration"])
    df["LaggedCEOPay"] = df.groupby("CompanyKey")["CEOPay"].shift(1)

    # Flag near-zero / negative equity, which can distort ROE (Section 3.3)
    df["equity_flag"] = df["Equity"] <= (0.05 * df["Total Assets"])

    # Use plain strings (not pandas 'category' dtype) for the year variable.
    # A categorical dtype retains unused categories after later filtering
    # (e.g. excluding 2020-2021 in the robustness check), which makes patsy
    # emit an all-zero dummy column and silently corrupts that year's
    # coefficient.
    df["Year_cat"] = df["Year"].astype(str)

    # "Company" is used throughout the rest of the pipeline (labels,
    # clustering, plots) -- prefer Sheet2's original display name, falling
    # back to the board name from Sheet1 if a row somehow lacks one.
    df["Company"] = df["Company"].fillna(df.get("BoardName"))

    # Missing-value handling (project preference): fill missing values in
    # the key modelling variables with 0 rather than dropping the row. This
    # replaces the previous approach of relying on dropna() at model-fitting
    # time. Each fill is counted and reported so the effect on the sample is
    # visible before trusting downstream results.
    for c in FILL_ZERO_COLS:
        n_missing = int(df[c].isna().sum())
        if n_missing:
            print(f"[data check] '{c}': {n_missing} missing value(s) filled with 0 "
                  f"(instead of dropping the row).")
        df[c] = df[c].fillna(0)

    return df


def winsorise(series: pd.Series, limits=WINSOR_LIMITS) -> pd.Series:
    lower = series.quantile(limits[0])
    upper = series.quantile(1 - limits[1])
    return series.clip(lower=lower, upper=upper)


def apply_winsorisation(df: pd.DataFrame, cols) -> pd.DataFrame:
    wdf = df.copy()
    for c in cols:
        wdf[c] = winsorise(wdf[c])
    return wdf


# ---------------------------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS  (Section 3.4)
# ---------------------------------------------------------------------------

def descriptive_statistics(df: pd.DataFrame, cols) -> pd.DataFrame:
    desc = df[cols].describe().T
    desc["skew"] = df[cols].skew()
    desc["kurtosis"] = df[cols].kurtosis()
    return desc


# ---------------------------------------------------------------------------
# 3. REGRESSION MODELS  (Section 3.4, 3.5)
# ---------------------------------------------------------------------------

def fit_pooled_ols(formula: str, data: pd.DataFrame, cluster_col: str = None):
    """Pooled OLS with HC1 robust SE, or firm-clustered SE if cluster_col given."""
    model = smf.ols(formula, data=data, missing="drop")
    if cluster_col is None:
        return model.fit(cov_type="HC1")
    return model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})


def fit_fixed_effects(formula: str, data: pd.DataFrame, entity_col: str = "CompanyKey",
                       cluster_col: str = "CompanyKey"):
    """Company fixed-effects model via LSDV (dummy variables), firm-clustered SE."""
    fe_formula = f"{formula} + C({entity_col})"
    model = smf.ols(fe_formula, data=data, missing="drop")
    return model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})


def wild_cluster_bootstrap_pvalue(formula: str, data: pd.DataFrame, param: str,
                                   cluster_col: str = "CompanyKey",
                                   n_boot: int = N_BOOTSTRAP) -> dict:
    """
    Wild-cluster (Rademacher) bootstrap p-value for a single coefficient,
    following the restricted (null-imposed) wild bootstrap of Cameron,
    Gelbach and Miller, appropriate with a small number of clusters
    (Abadie et al., 2023; MacKinnon, Nielsen and Webb, 2023).
    """
    model = smf.ols(formula, data=data, missing="drop")
    fitted = model.fit()
    used = fitted.model.data.frame  # rows actually used (after dropping NA)

    y, X = fitted.model.endog, fitted.model.exog
    exog_names = fitted.model.exog_names
    clusters = used[cluster_col].values
    unique_clusters = np.unique(clusters)

    if param not in exog_names:
        raise ValueError(f"{param} not found among regressors: {exog_names}")
    p_idx = exog_names.index(param)

    # Restricted model: impose beta_param = 0
    X_restricted = np.delete(X, p_idx, axis=1)
    restricted_fit = sm.OLS(y, X_restricted).fit()
    resid_restricted = np.zeros(len(y))
    resid_restricted[:] = y - X_restricted @ restricted_fit.params
    fitted_restricted = X_restricted @ restricted_fit.params

    full_fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    t_obs = full_fit.tvalues[p_idx]

    boot_t = np.empty(n_boot)
    for b in range(n_boot):
        weights = np.random.choice([-1.0, 1.0], size=len(unique_clusters))
        w_map = dict(zip(unique_clusters, weights))
        w = np.array([w_map[c] for c in clusters])
        y_star = fitted_restricted + resid_restricted * w
        boot_fit = sm.OLS(y_star, X).fit(cov_type="cluster", cov_kwds={"groups": clusters})
        boot_t[b] = boot_fit.tvalues[p_idx]

    p_value = np.mean(np.abs(boot_t) >= np.abs(t_obs))
    return {"t_observed": t_obs, "wild_cluster_p_value": p_value, "n_boot": n_boot}


# ---------------------------------------------------------------------------
# 4. DIAGNOSTICS  (Section 3.5)
# ---------------------------------------------------------------------------

def diagnostics(fitted_model, data: pd.DataFrame, exog_cols) -> dict:
    out = {}

    X = data[exog_cols].dropna()
    X_const = sm.add_constant(X)
    vif = pd.Series(
        [variance_inflation_factor(X_const.values, i) for i in range(1, X_const.shape[1])],
        index=exog_cols,
        name="VIF",
    )
    out["vif"] = vif

    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(fitted_model.resid, fitted_model.model.exog)
    out["breusch_pagan"] = {"LM_stat": bp_lm, "LM_p": bp_lm_p, "F_stat": bp_f, "F_p": bp_f_p}

    influence = OLSInfluence(fitted_model)
    cooks_d, _ = influence.cooks_distance
    out["cooks_distance_max"] = float(np.max(cooks_d))
    out["cooks_distance_over_threshold"] = int(np.sum(cooks_d > 4 / fitted_model.nobs))

    try:
        reset_res = linear_reset(fitted_model, power=2, use_f=True)
        out["ramsey_reset"] = {"F_stat": reset_res.fvalue, "p_value": reset_res.pvalue}
    except Exception as e:
        out["ramsey_reset"] = {"error": str(e)}

    return out


# ---------------------------------------------------------------------------
# 5. PLOTS  (values printed to terminal immediately before each plt.show())
# ---------------------------------------------------------------------------

def plot_histograms(df: pd.DataFrame, cols):
    print(f"\n{'-' * 90}\n[plot] Histogram input values (one row per firm-year)\n{'-' * 90}")
    print(df[["Company", "Year"] + list(cols)].to_string(index=False))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.flat, cols):
        data = df[c].dropna()
        ax.hist(data, bins=15, color="steelblue", edgecolor="black")
        ax.set_title(c)
        ax.set_xlabel(c)
        ax.set_ylabel("Frequency")
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    fig.suptitle("Distributions of key variables (Section 3.4)")
    plt.tight_layout()
    plt.show()


def plot_boxplots(df: pd.DataFrame, cols):
    print(f"\n{'-' * 90}\n[plot] Boxplot input values (one row per firm-year)\n{'-' * 90}")
    print(df[["Company", "Year"] + list(cols)].to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    data = [df[c].dropna().values for c in cols]
    ax.boxplot(data, tick_labels=list(cols), showmeans=True)
    ax.set_title("Boxplots of key variables -- extreme-value screening (Section 3.4)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.show()


def plot_scatter_pay_performance(df: pd.DataFrame):
    pairs = [("CEOPay", "ROA"), ("CEOPay", "ROE"),
             ("LaggedCEOPay", "ROA"), ("LaggedCEOPay", "ROE")]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (x, y) in zip(axes.flat, pairs):
        sub = df[["Company", "Year", x, y]].dropna()
        print(f"\n{'-' * 90}\n[plot] Scatter input values: {y} vs {x} (N={len(sub)})\n{'-' * 90}")
        print(sub.to_string(index=False))

        ax.scatter(sub[x], sub[y], alpha=0.7, color="steelblue", edgecolor="black")
        if len(sub) > 1:
            coeffs = np.polyfit(sub[x], sub[y], 1)
            xs = np.linspace(sub[x].min(), sub[x].max(), 50)
            ax.plot(xs, np.polyval(coeffs, xs), color="red", linestyle="--",
                    label=f"fit: y={coeffs[0]:.3f}x+{coeffs[1]:.3f}")
            ax.legend(fontsize=8)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y} vs {x}")
    fig.suptitle("CEO pay vs firm performance (raw firm-year observations)")
    plt.tight_layout()
    plt.show()


def plot_year_trends(df: pd.DataFrame):
    trend = df.groupby("Year")[["ROA", "ROE", "CEOPay"]].mean()
    print(f"\n{'-' * 90}\n[plot] Year-average values feeding the trend plot\n{'-' * 90}")
    print(trend)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, c in zip(axes, trend.columns):
        ax.plot(trend.index, trend[c], marker="o", color="darkorange")
        ax.set_title(f"Mean {c} by year")
        ax.set_xlabel("Year")
        ax.set_ylabel(c)
    fig.suptitle("Year-on-year trends (context for the year-dummy controls, Section 3.3)")
    plt.tight_layout()
    plt.show()


def plot_regression_diagnostics(fitted_model, model_df: pd.DataFrame, model_name: str):
    """Residuals-vs-fitted, Q-Q, and Cook's distance plots for one pooled-OLS
    (HC1) model -- the visual counterpart to Section 3.5's numeric diagnostics."""
    fitted_vals = fitted_model.fittedvalues
    resid = fitted_model.resid
    infl = OLSInfluence(fitted_model)
    cooks_d, _ = infl.cooks_distance
    threshold = 4 / fitted_model.nobs

    diag_df = pd.DataFrame({
        "fitted": fitted_vals.values,
        "residual": resid.values,
        "cooks_distance": cooks_d,
    })
    print(f"\n{'-' * 90}\n[plot] Diagnostic values for {model_name} "
          f"(Cook's distance threshold = 4/n = {threshold:.4f})\n{'-' * 90}")
    print(diag_df.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    axes[0].scatter(fitted_vals, resid, alpha=0.7, color="steelblue", edgecolor="black")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")

    qqplot(resid, line="45", ax=axes[1])
    axes[1].set_title("Normal Q-Q plot of residuals")

    axes[2].vlines(range(len(cooks_d)), 0, cooks_d, color="steelblue")
    axes[2].scatter(range(len(cooks_d)), cooks_d, color="steelblue", s=15)
    axes[2].axhline(threshold, color="red", linestyle="--", label="4/n threshold")
    axes[2].set_xlabel("Observation index")
    axes[2].set_ylabel("Cook's distance")
    axes[2].set_title("Influence (Cook's distance)")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"Regression diagnostics: {model_name}")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6. MAIN ANALYSIS PIPELINE
# ---------------------------------------------------------------------------

MODEL_SPECS = {
    "ROA_contemporaneous": ("ROA", "CEOPay"),
    "ROE_contemporaneous": ("ROE", "CEOPay"),
    "ROA_lagged": ("ROA", "LaggedCEOPay"),
    "ROE_lagged": ("ROE", "LaggedCEOPay"),
}


def build_formula(dep: str, pay_var: str) -> str:
    return f"{dep} ~ {pay_var} + FirmSize + Leverage + C(Year_cat)"


def run_all_models(df: pd.DataFrame, label: str, make_plots: bool = False):
    print(f"\n{'=' * 90}\n{label}\n{'=' * 90}")
    results = {}

    for name, (dep, pay_var) in MODEL_SPECS.items():
        formula = build_formula(dep, pay_var)
        # Missing-value handling (project preference): fill with 0 rather
        # than drop the row. By this point construct_variables() has
        # already filled the key columns, so this is a safety net only.
        model_df = df.copy()
        model_df[[dep, pay_var, "FirmSize", "Leverage"]] = (
            model_df[[dep, pay_var, "FirmSize", "Leverage"]].fillna(0)
        )

        print(f"\n--- {name}  ({formula}) ---")
        print(f"N = {len(model_df)}")

        pooled_hc1 = fit_pooled_ols(formula, model_df)
        pooled_cluster = fit_pooled_ols(formula, model_df, cluster_col="CompanyKey")
        fe_cluster = fit_fixed_effects(formula, model_df)

        print(f"\nPooled OLS (HC1 robust SE):")
        print(pooled_hc1.summary())

        print(f"\nPooled OLS (firm-clustered SE):")
        print(pooled_cluster.summary())

        print(f"\nCompany fixed effects (firm-clustered SE):")
        print(fe_cluster.summary())
        print(f"\nCompany fixed effects -- coefficient of interest ({pay_var}): "
              f"{fe_cluster.params.get(pay_var, np.nan):.6f}  "
              f"p = {fe_cluster.pvalues.get(pay_var, np.nan):.4f}")

        try:
            wcb = wild_cluster_bootstrap_pvalue(formula, model_df, pay_var, cluster_col="CompanyKey")
            print(f"\nWild-cluster bootstrap on {pay_var}: t = {wcb['t_observed']:.3f}, "
                  f"p = {wcb['wild_cluster_p_value']:.4f}  (B={wcb['n_boot']})")
        except Exception as e:
            wcb = {"error": str(e)}
            print(f"\nWild-cluster bootstrap failed: {e}")

        diag = diagnostics(pooled_hc1, model_df, [pay_var, "FirmSize", "Leverage"])
        print(f"\nVIF:\n{diag['vif']}")
        print(f"Breusch-Pagan: LM p = {diag['breusch_pagan']['LM_p']:.4f}")
        print(f"Max Cook's distance: {diag['cooks_distance_max']:.4f} "
              f"({diag['cooks_distance_over_threshold']} obs over 4/n threshold)")
        if "p_value" in diag["ramsey_reset"]:
            print(f"Ramsey RESET p = {diag['ramsey_reset']['p_value']:.4f}")

        if make_plots and SHOW_PLOTS:
            plot_regression_diagnostics(pooled_hc1, model_df, f"{label} -- {name}")

        results[name] = {
            "pooled_hc1": pooled_hc1,
            "pooled_cluster": pooled_cluster,
            "fe_cluster": fe_cluster,
            "wild_cluster_bootstrap": wcb,
            "diagnostics": diag,
            "n_obs": len(model_df),
        }

    return results


def main():
    panel = build_panel()
    df = construct_variables(panel)

    key_cols = ["ROA", "ROE", "CEOPay", "LaggedCEOPay", "FirmSize", "Leverage"]

    print(f"\nFinal panel: {df['CompanyKey'].nunique()} firms, {len(df)} firm-year observations, "
          f"{df['Year'].min()}-{df['Year'].max()}")

    desc = descriptive_statistics(df, key_cols)
    print("\n" + "=" * 90)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 90)
    print(desc)

    if SHOW_PLOTS:
        plot_histograms(df, key_cols)
        plot_boxplots(df, key_cols)
        plot_scatter_pay_performance(df)
        plot_year_trends(df)

    run_all_models(df, "Baseline (full sample)",
                    make_plots=(PLOT_DIAGNOSTICS_FOR == "Baseline (full sample)"))

    wdf = apply_winsorisation(df, ["ROA", "ROE", "CEOPay", "LaggedCEOPay", "FirmSize", "Leverage"])
    run_all_models(wdf, "Robustness: winsorised (1%/99%)",
                    make_plots=(PLOT_DIAGNOSTICS_FOR == "Robustness: winsorised (1%/99%)"))

    edf = df[~df["Year"].isin(EXCLUDE_YEARS)]
    run_all_models(edf, f"Robustness: excluding {EXCLUDE_YEARS}",
                    make_plots=(PLOT_DIAGNOSTICS_FOR == f"Robustness: excluding {EXCLUDE_YEARS}"))

    print("\nDone. All results and plot values were printed above; no files were written.")


if __name__ == "__main__":
    main()


import numpy as np
import pandas as pd
from colorama import init, Fore, Back, Style
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize colorama for colored terminal output
init(autoreset=True)

# Define the correlation matrix data
correlation_data = {
    'Variable': ['ROA', 'ROE', 'CEOPay', 'LaggedCEOPay', 'FirmSize', 'Leverage'],
    'ROA': [1.000000, 0.792530, -0.127397, -0.278831, -0.532967, -0.297376],
    'ROE': [0.792530, 1.000000, -0.120406, -0.224031, -0.424725, -0.123464],
    'CEOPay': [-0.127397, -0.120406, 1.000000, 0.479829, 0.281164, -0.090941],
    'LaggedCEOPay': [-0.278831, -0.224031, 0.479829, 1.000000, 0.319604, 0.018849],
    'FirmSize': [-0.532967, -0.424725, 0.281164, 0.319604, 1.000000, -0.086717],
    'Leverage': [-0.297376, -0.123464, -0.090941, 0.018849, -0.086717, 1.000000]
}

# Create DataFrame
df_corr = pd.DataFrame(correlation_data)
df_corr.set_index('Variable', inplace=True)

def color_correlation(value):
    """Return color based on correlation value strength"""
    abs_val = abs(value)
    if abs_val >= 0.7:
        return Back.RED + Fore.WHITE if value > 0 else Back.BLUE + Fore.WHITE
    elif abs_val >= 0.5:
        return Back.LIGHTRED_EX + Fore.BLACK if value > 0 else Back.LIGHTBLUE_EX + Fore.BLACK
    elif abs_val >= 0.3:
        return Back.YELLOW + Fore.BLACK if value > 0 else Back.CYAN + Fore.BLACK
    elif abs_val >= 0.1:
        return Back.LIGHTYELLOW_EX + Fore.BLACK if value > 0 else Back.LIGHTCYAN_EX + Fore.BLACK
    else:
        return Back.WHITE + Fore.BLACK

def print_heatmap(df):
    """Print a colored correlation heatmap in the terminal"""
    
    # Header
    print("\n" + "="*80)
    print(Fore.CYAN + Style.BRIGHT + "CORRELATION HEATMAP".center(80))
    print("="*80 + "\n")
    
    # Print column headers
    print(" " * 14, end="")
    for col in df.columns:
        print(f"{col:>12}", end="")
    print()
    
    # Print separator
    print(" " * 14 + "-" * (12 * len(df.columns)))
    
    # Print each row with colored cells
    for idx, row in df.iterrows():
        # Row label
        print(f"{idx:>14}", end=" ")
        
        for value in row:
            colored_value = color_correlation(value)
            # Format with 3 decimal places and padding
            formatted_val = f"{value:8.3f}"
            print(colored_value + formatted_val + Style.RESET_ALL, end="")
        
        print()  # New line after each row
    
    print("\n" + "="*80)
    print(Fore.GREEN + "Legend:")
    print(Fore.RED + "  ██" + Fore.WHITE + " Strong Positive (≥ 0.7)")
    print(Fore.LIGHTRED_EX + "  ██" + Fore.BLACK + " Moderate Positive (0.5-0.7)")
    print(Fore.YELLOW + "  ██" + Fore.BLACK + " Weak Positive (0.3-0.5)")
    print(Fore.LIGHTYELLOW_EX + "  ██" + Fore.BLACK + " Very Weak Positive (0.1-0.3)")
    print(Fore.WHITE + "  ██" + Fore.BLACK + " Very Weak/Negligible (< 0.1)")
    print(Fore.LIGHTCYAN_EX + "  ██" + Fore.BLACK + " Very Weak Negative (-0.1 to -0.3)")
    print(Fore.CYAN + "  ██" + Fore.BLACK + " Weak Negative (-0.3 to -0.5)")
    print(Fore.LIGHTBLUE_EX + "  ██" + Fore.BLACK + " Moderate Negative (-0.5 to -0.7)")
    print(Fore.BLUE + "  ██" + Fore.WHITE + " Strong Negative (≤ -0.7)")
    print("="*80 + "\n")

# Print the heatmap
print_heatmap(df_corr)

# Optional: Also display a numerical matrix for reference
print(Fore.CYAN + Style.BRIGHT + "\nNUMERICAL CORRELATION MATRIX")
print("-" * 80)
print(df_corr.round(3))
print("-" * 80)

# Create and display a matplotlib/seaborn heatmap (if environment supports it)
try:
    # Create a matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with annotations
    sns.heatmap(df_corr, 
                annot=True, 
                fmt='.3f', 
                cmap='RdBu_r', 
                center=0,
                vmin=-1, 
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 10})
    
    plt.title('Correlation Heatmap of Variables', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(Fore.GREEN + "\n✓ Heatmap saved as 'correlation_heatmap.png'")
    
    # Try to display the plot
    plt.show()
    
except Exception as e:
    print(Fore.YELLOW + f"\nNote: Matplotlib/Seaborn visualization not available: {e}")

# Print summary of key correlations
print(Fore.CYAN + Style.BRIGHT + "\nKEY CORRELATION INSIGHTS")
print("-" * 80)

insights = [
    ("ROA ↔ ROE", df_corr.loc["ROA", "ROE"], "Strong positive correlation (0.793) — firms with higher ROA tend to have higher ROE"),
    ("ROA ↔ FirmSize", df_corr.loc["ROA", "FirmSize"], "Moderate negative correlation (-0.533) — larger firms tend to have lower ROA"),
    ("FirmSize ↔ ROE", df_corr.loc["FirmSize", "ROE"], "Moderate negative correlation (-0.425) — larger firms tend to have lower ROE"),
    ("CEOPay ↔ LaggedCEOPay", df_corr.loc["CEOPay", "LaggedCEOPay"], "Moderate positive correlation (0.480) — CEO pay is persistent over time"),
    ("ROA ↔ Leverage", df_corr.loc["ROA", "Leverage"], "Weak negative correlation (-0.297) — higher leverage associated with lower ROA"),
    ("FirmSize ↔ CEOPay", df_corr.loc["FirmSize", "CEOPay"], "Weak positive correlation (0.281) — larger firms pay CEOs more"),
    ("CEO Pay ↔ Performance", df_corr.loc["CEOPay", "ROA"], "Very weak negative correlation (-0.127) — no strong linear relationship"),
    ("LaggedCEO Pay ↔ Performance", df_corr.loc["LaggedCEOPay", "ROA"], "Weak negative correlation (-0.279) — prior pay weakly associated with lower ROA")
]

for title, value, description in insights:
    color = Fore.GREEN if abs(value) > 0.3 else Fore.YELLOW
    print(f"{color}• {title}: {value:.3f} — {description}")

print("\n" + "="*80)
print(Fore.CYAN + "Correlation heatmap analysis complete!")
