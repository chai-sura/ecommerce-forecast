import pandas as pd
import numpy as np
from scipy import stats


def anova_test(df: pd.DataFrame, group_col: str, target_col: str) -> dict:
    """Run one-way ANOVA test for a grouping variable."""
    groups = [df[df[group_col] == g][target_col].values
              for g in df[group_col].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    significant   = p_val < 0.05
    print(f"=== ANOVA: {target_col} ~ {group_col} ===")
    print(f"F-statistic : {f_stat:.4f}")
    print(f"P-value     : {p_val:.4e}")
    print(f"→ {group_col} is {'a SIGNIFICANT' if significant else 'NOT a significant'} demand driver")
    return {"group_col": group_col, "f_stat": round(f_stat, 4),
            "p_value": round(p_val, 6), "significant": significant}


def price_elasticity(prices: pd.Series, quantities: pd.Series) -> dict:
    """Compute log-log price elasticity coefficient."""
    log_price = np.log(prices + 1)
    log_qty   = np.log(quantities + 1)
    slope, intercept, r, p, se = stats.linregress(log_price, log_qty)
    print(f"Elasticity Coefficient : {slope:.4f}")
    print(f"R-squared              : {r**2:.4f}")
    print(f"P-value                : {p:.4f}")
    if slope < -1:
        print("→ ELASTIC: customers are price-sensitive")
    elif slope < 0:
        print("→ INELASTIC: modest price sensitivity")
    else:
        print("→ Positive relationship detected")
    return {"elasticity": round(slope, 4), "r_squared": round(r**2, 4),
            "p_value": round(p, 6), "std_error": round(se, 4)}


def promotional_lift(df: pd.DataFrame,
                     flag_col: str,
                     target_col: str = "sales") -> dict:
    """Compute promotional lift and t-test between two groups."""
    group_1 = df[df[flag_col] == 1][target_col]
    group_0 = df[df[flag_col] == 0][target_col]
    t_stat, p_val = stats.ttest_ind(group_1, group_0)
    lift = (group_1.mean() - group_0.mean()) / group_0.mean() * 100
    print(f"=== Promotional Lift: {flag_col} ===")
    print(f"Group 1 avg  : {group_1.mean():.2f}")
    print(f"Group 0 avg  : {group_0.mean():.2f}")
    print(f"Lift         : +{lift:.1f}%")
    print(f"T-statistic  : {t_stat:.4f}")
    print(f"P-value      : {p_val:.4e}")
    print(f"→ Lift is {'STATISTICALLY SIGNIFICANT' if p_val < 0.05 else 'not significant'}")
    return {"flag_col": flag_col, "lift_pct": round(lift, 2),
            "t_stat": round(t_stat, 4), "p_value": round(p_val, 6),
            "significant": p_val < 0.05}


def ols_regression(df: pd.DataFrame,
                   features: list,
                   target_col: str = "sales"):
    """Fit OLS regression and return fitted model."""
    import statsmodels.api as sm
    X     = sm.add_constant(df[features])
    y     = df[target_col]
    model = sm.OLS(y, X).fit()
    return model


def significant_features(model, alpha: float = 0.05) -> pd.DataFrame:
    """Extract significant features from a fitted OLS model."""
    pvals  = model.pvalues.drop("const")
    coefs  = model.params.drop("const")
    result = pd.DataFrame({"feature": pvals.index,
                            "coefficient": coefs.values,
                            "p_value": pvals.values})
    result["significant"] = result["p_value"] < alpha
    result = result.sort_values("p_value")
    sig   = result[result["significant"]]
    insig = result[~result["significant"]]
    print(f"✅ Significant ({len(sig)}): {sig['feature'].tolist()}")
    print(f"❌ Not significant ({len(insig)}): {insig['feature'].tolist()}")
    print(f"\nR²     : {model.rsquared:.4f}")
    print(f"Adj R² : {model.rsquared_adj:.4f}")
    return result


def ridge_lasso_cv(df: pd.DataFrame, features: list,
                   target_col: str = "sales", cv: int = 5) -> dict:
    """Fit Ridge and Lasso with cross-validation, return CV R² scores."""
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    y        = df[target_col].values

    ridge    = Ridge(alpha=1.0)
    lasso    = Lasso(alpha=0.1)
    ridge_cv = cross_val_score(ridge, X_scaled, y, cv=cv, scoring="r2")
    lasso_cv = cross_val_score(lasso, X_scaled, y, cv=cv, scoring="r2")

    print(f"Ridge CV R² : {ridge_cv.mean():.4f} ± {ridge_cv.std():.4f}")
    print(f"Lasso CV R² : {lasso_cv.mean():.4f} ± {lasso_cv.std():.4f}")

    lasso.fit(X_scaled, y)
    zeroed = [f for f, c in zip(features, lasso.coef_) if c == 0]
    print(f"Lasso zeroed: {zeroed}")

    return {"ridge_r2": round(ridge_cv.mean(), 4),
            "lasso_r2": round(lasso_cv.mean(), 4),
            "lasso_zeroed": zeroed}


def adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller stationarity test."""
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series)
    stationary = result[1] < 0.05
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"P-value       : {result[1]:.4f}")
    print(f"→ Series is {'STATIONARY' if stationary else 'NON-STATIONARY'}")
    return {"adf_stat": round(result[0], 4), "p_value": round(result[1], 6),
            "stationary": stationary}


if __name__ == "__main__":
    print("Evaluation module loaded ✅")
    print("Available: anova_test, price_elasticity, promotional_lift,")
    print("           ols_regression, significant_features, ridge_lasso_cv, adf_test")