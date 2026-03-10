import pandas as pd
import numpy as np
from scipy import stats


def compute_item_stats(demand: pd.DataFrame,
                       item_col: str = "item",
                       target_col: str = "sales") -> pd.DataFrame:
    """Compute per-item demand statistics."""
    stats_df = demand.groupby(item_col).agg(
        avg_daily_demand = (target_col, "mean"),
        std_daily_demand = (target_col, "std"),
        max_daily_demand = (target_col, "max"),
        min_daily_demand = (target_col, "min"),
        total_sales      = (target_col, "sum"),
        days_active      = (target_col, "count")
    ).reset_index()
    stats_df["cv"] = stats_df["std_daily_demand"] / stats_df["avg_daily_demand"]
    return stats_df


def compute_safety_stock(std_demand: pd.Series,
                          lead_time: int = 7,
                          service_level: float = 0.95) -> pd.Series:
    """
    Compute safety stock per item.
    Safety Stock = Z * std_demand * sqrt(lead_time)
    """
    z_score = stats.norm.ppf(service_level)
    return (z_score * std_demand * np.sqrt(lead_time)).round(2)


def compute_rop(avg_demand: pd.Series,
                safety_stock: pd.Series,
                lead_time: int = 7) -> pd.Series:
    """
    Compute Reorder Point (ROP) per item.
    ROP = avg_daily_demand * lead_time + safety_stock
    """
    return (avg_demand * lead_time + safety_stock).round(2)


def compute_eoq(avg_demand: pd.Series,
                ordering_cost: float = 50.0,
                holding_cost: float = 2.0) -> pd.Series:
    """
    Compute Economic Order Quantity (EOQ) per item.
    EOQ = sqrt(2 * annual_demand * ordering_cost / holding_cost)
    """
    annual_demand = avg_demand * 365
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    return eoq.round(2)


def compute_inventory_cost(avg_demand: pd.Series,
                            eoq: pd.Series,
                            ordering_cost: float = 50.0,
                            holding_cost: float = 2.0) -> pd.Series:
    """Compute annual inventory cost under EOQ policy."""
    annual_demand = avg_demand * 365
    cost = (annual_demand / eoq) * ordering_cost + (eoq / 2) * holding_cost
    return cost.round(2)


def compute_baseline_cost(avg_demand: pd.Series,
                           order_freq: int = 12,
                           ordering_cost: float = 50.0,
                           holding_cost: float = 2.0) -> pd.Series:
    """Compute annual inventory cost under baseline fixed-interval ordering."""
    annual_demand = avg_demand * 365
    cost = order_freq * ordering_cost + (annual_demand / order_freq / 2) * holding_cost
    return cost.round(2)


def compute_stockout_risk(demand: pd.DataFrame,
                           item_stats: pd.DataFrame,
                           item_col: str = "item",
                           target_col: str = "sales") -> pd.DataFrame:
    """
    Compute stockout risk % per item.
    Risk = % of days demand exceeded avg + safety_stock threshold.
    """
    results = []
    for item_id in demand[item_col].unique():
        item_demand = demand[demand[item_col] == item_id][target_col].values
        row         = item_stats[item_stats[item_col] == item_id].iloc[0]
        threshold   = row["avg_daily_demand"] + row["safety_stock"]
        stockout_days = np.sum(item_demand > threshold)
        stockout_pct  = stockout_days / len(item_demand) * 100
        results.append({item_col: item_id,
                         "stockout_days": stockout_days,
                         "stockout_risk": round(stockout_pct, 2)})
    return pd.DataFrame(results)


def get_recommendation(row: pd.Series) -> str:
    """Generate inventory recommendation for an item."""
    if row["stockout_risk"] > 15:
        return "🔴 URGENT — Increase safety stock immediately"
    elif row["stockout_risk"] > 10:
        return "🟡 WARNING — Review reorder point"
    elif row["cv"] > 0.3:
        return "🟠 MONITOR — High demand variability"
    elif row["savings_pct"] > 20:
        return "💰 OPTIMIZE — Switch to EOQ ordering"
    else:
        return "🟢 HEALTHY — No action needed"


def run_supply_chain_pipeline(demand: pd.DataFrame,
                               lead_time: int = 7,
                               service_level: float = 0.95,
                               ordering_cost: float = 50.0,
                               holding_cost: float = 2.0) -> pd.DataFrame:
    """
    Full supply chain pipeline — returns complete item-level results.
    Runs: item stats → safety stock → ROP → EOQ → stockout risk → recommendations.
    """
    print(f"Running supply chain pipeline...")
    print(f"  Lead time     : {lead_time} days")
    print(f"  Service level : {service_level*100:.0f}%")
    print(f"  Ordering cost : ${ordering_cost}")
    print(f"  Holding cost  : ${holding_cost}/unit/year")
    print()

    item_stats = compute_item_stats(demand)
    item_stats["safety_stock"]   = compute_safety_stock(
        item_stats["std_daily_demand"], lead_time, service_level)
    item_stats["rop"]            = compute_rop(
        item_stats["avg_daily_demand"], item_stats["safety_stock"], lead_time)
    item_stats["annual_demand"]  = item_stats["avg_daily_demand"] * 365
    item_stats["eoq"]            = compute_eoq(
        item_stats["avg_daily_demand"], ordering_cost, holding_cost)
    item_stats["orders_per_year"] = (item_stats["annual_demand"] / item_stats["eoq"]).round(1)
    item_stats["annual_inv_cost"] = compute_inventory_cost(
        item_stats["avg_daily_demand"], item_stats["eoq"], ordering_cost, holding_cost)
    item_stats["baseline_cost"]  = compute_baseline_cost(
        item_stats["avg_daily_demand"], 12, ordering_cost, holding_cost)
    item_stats["cost_savings"]   = (item_stats["baseline_cost"] - item_stats["annual_inv_cost"]).round(2)
    item_stats["savings_pct"]    = (item_stats["cost_savings"] / item_stats["baseline_cost"] * 100).round(1)

    stockout_df = compute_stockout_risk(demand, item_stats)
    item_stats  = item_stats.merge(stockout_df, on="item")
    item_stats["recommendation"] = item_stats.apply(get_recommendation, axis=1)

    total_savings = item_stats["cost_savings"].sum()
    savings_pct   = total_savings / item_stats["baseline_cost"].sum() * 100
    high_risk     = len(item_stats[item_stats["stockout_risk"] > 10])

    print(f"✅ Pipeline complete")
    print(f"   Total items          : {len(item_stats)}")
    print(f"   High stockout risk   : {high_risk} items")
    print(f"   Total cost savings   : ${total_savings:,.2f}/year ({savings_pct:.1f}%)")

    return item_stats


if __name__ == "__main__":
    print("Supply chain module loaded ✅")
    print("Available: compute_item_stats, compute_safety_stock, compute_rop,")
    print("           compute_eoq, compute_stockout_risk, run_supply_chain_pipeline")