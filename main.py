import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path

from src.data_processing import (
    load_demand, load_retail, add_time_features,
    aggregate_daily, train_test_split_time, save_processed
)
from src.forecasting import (
    fit_sarima, fit_ets, fit_prophet,
    make_xgb_features, fit_xgboost,
    rolling_cross_validation, evaluate_forecast, XGB_FEATURES
)
from src.evaluation import (
    anova_test, price_elasticity, promotional_lift,
    ols_regression, significant_features, adf_test
)
from src.supply_chain import run_supply_chain_pipeline

DATA_PROCESSED = Path("data/processed")


def run_pipeline():
    print("=" * 55)
    print("  E-Commerce Demand & Supply Chain Pipeline")
    print("=" * 55)

    # ── 1. Load & Process Data ─────────────────────────────
    print("\n[1/4] Loading & processing data...")
    demand = load_demand()
    retail = load_retail()
    demand = add_time_features(demand)
    save_processed(demand, "demand_clean.csv")
    save_processed(retail, "retail_clean.csv")
    print(f"      Demand: {demand.shape} | Retail: {retail.shape}")

    # ── 2. Forecasting ─────────────────────────────────────
    print("\n[2/4] Running forecasting models...")
    daily        = aggregate_daily(demand)
    train, test  = train_test_split_time(daily, test_days=90)

    results = []

    # ADF stationarity check
    print("\n  ADF Test:")
    adf_test(train["sales"])

    # SARIMA
    print("\n  SARIMA:")
    sarima_pred = fit_sarima(train["sales"], steps=len(test))
    results.append(evaluate_forecast(test["sales"].values, sarima_pred, "SARIMA"))

    # ETS
    print("\n  ETS:")
    ets_pred = fit_ets(train["sales"], steps=len(test))
    results.append(evaluate_forecast(test["sales"].values, ets_pred, "ETS"))

    # Prophet
    print("\n  Prophet:")
    prophet_train = train.rename(columns={"date": "ds", "sales": "y"})
    prophet_pred, prophet_model, forecast = fit_prophet(prophet_train, steps=len(test))
    results.append(evaluate_forecast(test["sales"].values, prophet_pred, "Prophet"))

    # XGBoost
    print("\n  XGBoost:")
    full_df      = make_xgb_features(daily)
    full_clean   = full_df.dropna(subset=XGB_FEATURES)
    split_date   = train["date"].max()
    xgb_train    = full_clean[full_clean["date"] <= split_date]
    xgb_test     = full_clean[full_clean["date"] >  split_date]
    xgb_pred, xgb_model = fit_xgboost(xgb_train, xgb_test)
    results.append(evaluate_forecast(xgb_test["sales"].values, xgb_pred, "XGBoost"))

    # Rolling CV
    print("\n  Rolling Cross-Validation (XGBoost):")
    cv_scores = rolling_cross_validation(full_clean)

    # Save results
    results_df = pd.DataFrame(results).sort_values("RMSE")
    save_processed(results_df, "model_comparison.csv")
    best = results_df.iloc[0]
    print(f"\n  🏆 Best model: {best['model']} (RMSE: {best['RMSE']})")

    # Save forecast results
    test_out              = test.copy()
    test_out["sarima_pred"]  = sarima_pred
    test_out["ets_pred"]     = ets_pred
    test_out["prophet_pred"] = prophet_pred
    xgb_out = xgb_test[["date", "sales"]].copy()
    xgb_out["xgb_pred"] = xgb_pred
    forecast_out = test_out.merge(xgb_out[["date", "xgb_pred"]], on="date", how="left")
    save_processed(forecast_out, "forecast_results.csv")

    # ── 3. Regression Analysis ─────────────────────────────
    print("\n[3/4] Running regression analysis...")
    reg_features = ["month", "dow", "quarter", "is_weekend",
                    "is_holiday_season", "is_summer",
                    "lag_7", "lag_14", "roll_7", "roll_30"]

    # Add lag features to demand
    daily_agg = aggregate_daily(demand)
    daily_agg["lag_7"]  = daily_agg["sales"].shift(7)
    daily_agg["lag_14"] = daily_agg["sales"].shift(14)
    daily_agg["roll_7"] = daily_agg["sales"].shift(1).rolling(7).mean()
    daily_agg["roll_30"]= daily_agg["sales"].shift(1).rolling(30).mean()
    demand_reg = demand.merge(
        daily_agg[["date", "lag_7", "lag_14", "roll_7", "roll_30"]], on="date", how="left"
    ).dropna()

    print("\n  ANOVA Tests:")
    anova_test(demand_reg, "month", "sales")
    anova_test(demand_reg, "dow",   "sales")
    anova_test(demand_reg, "store", "sales")

    print("\n  OLS Regression:")
    ols_model = ols_regression(demand_reg, reg_features)
    sig_df    = significant_features(ols_model)
    save_processed(sig_df, "regression_summary.csv")

    print("\n  Promotional Lift (Holiday Season):")
    promotional_lift(demand_reg, "is_holiday_season", "sales")

    print("\n  Price Elasticity:")
    daily_retail = retail.groupby("date").agg(
        avg_price=("Price", "mean"), total_qty=("Quantity", "sum")
    ).reset_index()
    daily_retail = daily_retail[
        daily_retail["avg_price"] < daily_retail["avg_price"].quantile(0.95)
    ]
    price_elasticity(daily_retail["avg_price"], daily_retail["total_qty"])

    # ── 4. Supply Chain Optimization ──────────────────────
    print("\n[4/4] Running supply chain optimization...")
    item_stats = run_supply_chain_pipeline(demand)
    save_processed(item_stats, "supply_chain_results.csv")

    reco_cols  = ["item", "avg_daily_demand", "safety_stock", "rop",
                  "eoq", "stockout_risk", "cost_savings", "recommendation"]
    save_processed(item_stats[reco_cols], "inventory_recommendations.csv")

    # ── MLflow Logging ─────────────────────────────────────
    print("\n  Logging to MLflow...")
    mlflow.set_experiment("ecommerce_forecasting")
    for res in results:
        with mlflow.start_run(run_name=res["model"]):
            mlflow.log_param("model",  res["model"])
            mlflow.log_metric("RMSE",  res["RMSE"])
            mlflow.log_metric("MAE",   res["MAE"])
            mlflow.log_metric("MAPE",  res["MAPE"])
    with mlflow.start_run(run_name="XGBoost_final"):
        mlflow.log_param("n_estimators",  500)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("max_depth",     4)
        mlflow.log_metric("RMSE",         results[-1]["RMSE"])
        mlflow.log_metric("cv_mean_rmse", round(np.mean(cv_scores), 2))
        mlflow.sklearn.log_model(xgb_model, "xgboost_model")

    # ── Final Summary ──────────────────────────────────────
    total_savings = item_stats["cost_savings"].sum()
    savings_pct   = total_savings / item_stats["baseline_cost"].sum() * 100

    print("\n" + "=" * 55)
    print("  PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Best model           : {best['model']} (RMSE: {best['RMSE']})")
    print(f"  CV Mean RMSE         : {np.mean(cv_scores):.2f}")
    print(f"  Significant features : {len(sig_df[sig_df['significant']])}")
    print(f"  Total items          : {len(item_stats)}")
    print(f"  Inventory savings    : ${total_savings:,.2f}/year ({savings_pct:.1f}%)")
    print(f"  High stockout items  : {len(item_stats[item_stats['stockout_risk'] > 10])}")
    print("=" * 55)
    print("\n  Run: mlflow ui   to view experiment dashboard")


if __name__ == "__main__":
    run_pipeline()