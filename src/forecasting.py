import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray, model_name: str) -> dict:
    """Compute RMSE, MAE, MAPE for a forecast."""
    actual    = np.array(actual)
    predicted = np.maximum(np.array(predicted), 0)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-9))) * 100
    print(f"--- {model_name} ---")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  MAPE : {mape:.2f}%")
    return {"model": model_name, "RMSE": round(rmse, 2),
            "MAE": round(mae, 2), "MAPE": round(mape, 2)}


def fit_sarima(train: pd.Series, steps: int,
               order=(1,1,1), seasonal_order=(1,1,1,7)):
    """Fit SARIMA model and return forecast array."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit   = model.fit(disp=False)
    preds = fit.forecast(steps=steps)
    return np.maximum(preds.values, 0)


def fit_ets(train: pd.Series, steps: int, seasonal_periods: int = 7):
    """Fit Holt-Winters ETS model and return forecast array."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(train, trend="add",
                                 seasonal="add",
                                 seasonal_periods=seasonal_periods)
    fit   = model.fit(optimized=True)
    preds = fit.forecast(steps=steps)
    return np.maximum(preds.values, 0)


def fit_prophet(train: pd.DataFrame, steps: int,
                yearly_seasonality: bool = True,
                weekly_seasonality: bool = True):
    """
    Fit Prophet model and return forecast array.
    train must have columns: ds (date), y (target)
    """
    from prophet import Prophet
    model = Prophet(yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05)
    model.fit(train)
    future   = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    preds    = forecast.tail(steps)["yhat"].values
    return np.maximum(preds, 0), model, forecast


def make_xgb_features(df: pd.DataFrame,
                       target_col: str = "sales") -> pd.DataFrame:
    """Add XGBoost time + lag features to daily dataframe."""
    df = df.copy()
    df["dayofweek"]  = df["date"].dt.dayofweek
    df["month"]      = df["date"].dt.month
    df["quarter"]    = df["date"].dt.quarter
    df["year"]       = df["date"].dt.year
    df["dayofyear"]  = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["lag_7"]      = df[target_col].shift(7)
    df["lag_14"]     = df[target_col].shift(14)
    df["lag_30"]     = df[target_col].shift(30)
    df["roll_7"]     = df[target_col].shift(1).rolling(7).mean()
    df["roll_30"]    = df[target_col].shift(1).rolling(30).mean()
    return df


XGB_FEATURES = ["dayofweek", "month", "quarter", "year",
                 "dayofyear", "weekofyear",
                 "lag_7", "lag_14", "lag_30", "roll_7", "roll_30"]


def fit_xgboost(train: pd.DataFrame, test: pd.DataFrame,
                features: list = None,
                n_estimators: int = 500,
                learning_rate: float = 0.05,
                max_depth: int = 4):
    """Fit XGBoost model and return predictions + model."""
    from xgboost import XGBRegressor
    features = features or XGB_FEATURES
    model = XGBRegressor(n_estimators=n_estimators,
                         learning_rate=learning_rate,
                         max_depth=max_depth,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         random_state=42)
    model.fit(train[features], train["sales"],
              eval_set=[(test[features], test["sales"])],
              verbose=False)
    preds = model.predict(test[features])
    return np.maximum(preds, 0), model


def rolling_cross_validation(df: pd.DataFrame,
                              features: list = None,
                              n_splits: int = 5) -> list:
    """Run TimeSeriesSplit cross-validation on XGBoost."""
    from xgboost import XGBRegressor
    features = features or XGB_FEATURES
    tscv     = TimeSeriesSplit(n_splits=n_splits)
    X        = df[features].values
    y        = df["sales"].values
    scores   = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        m = XGBRegressor(n_estimators=300, learning_rate=0.05,
                         max_depth=4, random_state=42)
        m.fit(X[tr_idx], y[tr_idx], verbose=False)
        preds = m.predict(X[val_idx])
        rmse  = np.sqrt(mean_squared_error(y[val_idx], preds))
        scores.append(rmse)
        print(f"  Fold {fold+1} RMSE: {rmse:.2f}")
    print(f"\nMean CV RMSE : {np.mean(scores):.2f}")
    print(f"Std  CV RMSE : {np.std(scores):.2f}")
    return scores


if __name__ == "__main__":
    print("Forecasting module loaded ✅")
    print("Available functions: fit_sarima, fit_ets, fit_prophet, fit_xgboost, rolling_cross_validation")