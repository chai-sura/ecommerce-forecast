import pandas as pd
import numpy as np
from pathlib import Path


DATA_RAW       = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"


def load_demand(filepath: str = None) -> pd.DataFrame:
    """Load and parse store demand dataset."""
    path = filepath or DATA_RAW / "store_demand.csv"
    df   = pd.read_csv(path, parse_dates=["date"])
    df   = df.sort_values("date").reset_index(drop=True)
    return df


def load_retail(filepath: str = None) -> pd.DataFrame:
    """Load and clean online retail dataset."""
    path   = filepath or DATA_RAW / "online_retail.csv"
    df     = pd.read_csv(path)
    df     = df.dropna(subset=["Customer ID"])
    df     = df[df["Quantity"] > 0]
    df     = df[df["Price"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], infer_datetime_format=True)
    df["Revenue"]     = df["Quantity"] * df["Price"]
    df["date"]        = df["InvoiceDate"].dt.normalize()
    df["month"]       = df["InvoiceDate"].dt.month
    df["year"]        = df["InvoiceDate"].dt.year
    return df


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add time-based features to a dataframe."""
    df = df.copy()
    df["year"]       = df[date_col].dt.year
    df["month"]      = df[date_col].dt.month
    df["dow"]        = df[date_col].dt.dayofweek
    df["week"]       = df[date_col].dt.isocalendar().week.astype(int)
    df["quarter"]    = df[date_col].dt.quarter
    df["dayofyear"]  = df[date_col].dt.dayofyear
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)
    df["is_summer"]  = df["month"].isin([6, 7, 8]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "sales",
                     lags: list = [7, 14, 30],
                     rolling_windows: list = [7, 30]) -> pd.DataFrame:
    """Add lag and rolling mean features."""
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    for w in rolling_windows:
        df[f"roll_{w}"] = df[target_col].shift(1).rolling(w).mean()
    return df


def aggregate_daily(df: pd.DataFrame,
                    date_col: str = "date",
                    target_col: str = "sales") -> pd.DataFrame:
    """Aggregate demand to daily total."""
    daily = df.groupby(date_col)[target_col].sum().reset_index()
    daily = daily.sort_values(date_col).reset_index(drop=True)
    return daily


def train_test_split_time(df: pd.DataFrame,
                           date_col: str = "date",
                           test_days: int = 90):
    """Split dataframe into train/test by date."""
    split_date = df[date_col].max() - pd.Timedelta(days=test_days)
    train = df[df[date_col] <= split_date].copy()
    test  = df[df[date_col] >  split_date].copy()
    return train, test


def save_processed(df: pd.DataFrame, filename: str):
    """Save dataframe to processed data directory."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / filename
    df.to_csv(path, index=False)
    print(f"✅ Saved: {path}")


if __name__ == "__main__":
    demand = load_demand()
    retail = load_retail()
    demand = add_time_features(demand)
    save_processed(demand, "demand_clean.csv")
    save_processed(retail, "retail_clean.csv")
    print(f"Demand: {demand.shape} | Retail: {retail.shape}")