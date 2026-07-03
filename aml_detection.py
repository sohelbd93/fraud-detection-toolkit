"""
aml_detection.py

Anti-Money Laundering (AML) transaction monitoring toolkit.

Combines classic AML red-flag rules (structuring/smurtping, rapid
in-and-out "layering" flows, velocity spikes, round-tripping between
accounts) with an unsupervised anomaly model (Isolation Forest) over
engineered account/transaction features, producing a blended risk score
per transaction suitable for analyst triage or SAR (Suspicious Activity
Report) candidate generation.

This is a research/demo-grade reference implementation trained and
evaluated on synthetic data. It is not a certified AML compliance system
and does not replace a licensed BSA/AML program.

Usage:
    python aml_detection.py --demo          # run on generated synthetic data
    python aml_detection.py --input tx.csv  # score your own transaction CSV
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# Rule-based red flags
# ----------------------------------------------------------------------

STRUCTURING_THRESHOLD = 10_000       # classic CTR reporting threshold (USD)
STRUCTURING_MARGIN = 0.10            # flag deposits just under threshold
VELOCITY_WINDOW_HOURS = 24
VELOCITY_TX_COUNT_FLAG = 8           # >N transactions per account per window
ROUND_TRIP_WINDOW_HOURS = 72


@dataclass
class AMLFlags:
    structuring: bool
    high_velocity: bool
    round_number: bool
    rapid_movement: bool

    def any_triggered(self) -> bool:
        return any([self.structuring, self.high_velocity, self.round_number, self.rapid_movement])


def flag_structuring(amount: float) -> bool:
    """Deposits just under the CTR threshold, a classic 'smurfing' pattern."""
    lower = STRUCTURING_THRESHOLD * (1 - STRUCTURING_MARGIN)
    return lower <= amount < STRUCTURING_THRESHOLD


def flag_round_number(amount: float) -> bool:
    """Suspiciously round amounts (e.g. exactly $5,000) are mildly indicative
    of manual, deliberate structuring rather than organic commerce."""
    return amount >= 1000 and amount % 500 == 0


def flag_high_velocity(df: pd.DataFrame, account_col: str = "account_id",
                        ts_col: str = "timestamp") -> pd.Series:
    """Flag transactions belonging to an account that exceeds a transaction
    count threshold within a rolling window."""
    df = df.sort_values(ts_col)
    flags = pd.Series(False, index=df.index)
    for acct, group in df.groupby(account_col):
        counts = group.set_index(ts_col)["amount"].rolling(f"{VELOCITY_WINDOW_HOURS}h").count()
        flagged_ts = counts[counts > VELOCITY_TX_COUNT_FLAG].index
        flags.loc[group[group[ts_col].isin(flagged_ts)].index] = True
    return flags


def flag_rapid_movement(df: pd.DataFrame, account_col: str = "account_id",
                         counterparty_col: str = "counterparty_id",
                         ts_col: str = "timestamp") -> pd.Series:
    """Flag accounts that receive funds and move them out again to a
    different counterparty within a short window ('layering')."""
    df = df.sort_values(ts_col)
    flags = pd.Series(False, index=df.index)
    for acct, group in df.groupby(account_col):
        inflows = group[group["direction"] == "in"]
        outflows = group[group["direction"] == "out"]
        for _, inflow in inflows.iterrows():
            window_end = inflow[ts_col] + pd.Timedelta(hours=ROUND_TRIP_WINDOW_HOURS)
            matching_out = outflows[
                (outflows[ts_col] > inflow[ts_col]) & (outflows[ts_col] <= window_end)
            ]
            if not matching_out.empty:
                flags.loc[inflow.name] = True
                flags.loc[matching_out.index] = True
    return flags


def apply_rule_engine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["flag_structuring"] = df["amount"].apply(flag_structuring)
    df["flag_round_number"] = df["amount"].apply(flag_round_number)
    df["flag_high_velocity"] = flag_high_velocity(df)
    df["flag_rapid_movement"] = flag_rapid_movement(df)
    df["rule_flag_count"] = df[
        ["flag_structuring", "flag_round_number", "flag_high_velocity", "flag_rapid_movement"]
    ].sum(axis=1)
    return df


# ----------------------------------------------------------------------
# Unsupervised anomaly scoring (Isolation Forest)
# ----------------------------------------------------------------------

FEATURE_COLUMNS = ["amount", "rule_flag_count"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    acct_mean = df.groupby("account_id")["amount"].transform("mean")
    acct_std = df.groupby("account_id")["amount"].transform("std")
    df["amount_zscore_vs_account"] = (
        (df["amount"] - acct_mean) / acct_std.replace(0, np.nan)
    ).fillna(0)
    return df


def score_anomalies(df: pd.DataFrame, contamination: float = 0.03, seed: int = 42) -> pd.DataFrame:
    df = engineer_features(df)
    features = df[FEATURE_COLUMNS + ["hour_of_day", "amount_zscore_vs_account"]].fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    model.fit(X)

    # decision_function: higher = more normal. Invert + rescale to [0, 1] risk score.
    raw_scores = model.decision_function(X)
    df["anomaly_score"] = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min() + 1e-9)
    return df


def blended_risk_score(df: pd.DataFrame, rule_weight: float = 0.4, model_weight: float = 0.6) -> pd.DataFrame:
    df = df.copy()
    normalized_rule_count = df["rule_flag_count"] / df["rule_flag_count"].max() if df["rule_flag_count"].max() > 0 else 0
    df["risk_score"] = rule_weight * normalized_rule_count + model_weight * df["anomaly_score"]
    df["sar_candidate"] = df["risk_score"] >= df["risk_score"].quantile(0.97)
    return df


# ----------------------------------------------------------------------
# Synthetic demo data
# ----------------------------------------------------------------------

def generate_synthetic_transactions(n_accounts: int = 50, n_transactions: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    account_ids = [f"ACCT-{i:04d}" for i in range(n_accounts)]

    start = pd.Timestamp("2026-01-01")
    timestamps = start + pd.to_timedelta(rng.integers(0, 60 * 24 * 60, n_transactions), unit="m")

    df = pd.DataFrame({
        "transaction_id": [f"TX-{i:06d}" for i in range(n_transactions)],
        "account_id": rng.choice(account_ids, n_transactions),
        "counterparty_id": rng.choice(account_ids, n_transactions),
        "timestamp": timestamps,
        "direction": rng.choice(["in", "out"], n_transactions),
        "amount": np.round(rng.lognormal(mean=5.5, sigma=1.2, size=n_transactions), 2),
    })

    # inject a handful of structuring patterns
    structuring_idx = rng.choice(df.index, size=30, replace=False)
    df.loc[structuring_idx, "amount"] = rng.uniform(9000, 9900, size=len(structuring_idx))

    return df.sort_values("timestamp").reset_index(drop=True)


def run_demo() -> None:
    print("Generating synthetic transaction data...")
    df = generate_synthetic_transactions()

    print("Applying AML rule engine...")
    df = apply_rule_engine(df)

    print("Scoring with Isolation Forest...")
    df = score_anomalies(df)
    df = blended_risk_score(df)

    top = df.sort_values("risk_score", ascending=False).head(15)
    print(f"\nTotal transactions: {len(df)}")
    print(f"SAR candidates (top ~3% risk score): {df['sar_candidate'].sum()}")
    print("\nTop 15 highest-risk transactions:")
    print(top[["transaction_id", "account_id", "amount", "rule_flag_count",
               "anomaly_score", "risk_score", "sar_candidate"]].to_string(index=False))


def score_csv(path: str) -> None:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"transaction_id", "account_id", "counterparty_id", "timestamp", "direction", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input CSV missing required columns: {missing}")

    df = apply_rule_engine(df)
    df = score_anomalies(df)
    df = blended_risk_score(df)

    out_path = path.replace(".csv", "_scored.csv")
    df.to_csv(out_path, index=False)
    print(f"Scored {len(df)} transactions -> {out_path}")
    print(f"SAR candidates flagged: {df['sar_candidate'].sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Run on generated synthetic data")
    parser.add_argument("--input", type=str, help="Path to a transaction CSV to score")
    args = parser.parse_args()

    if args.input:
        score_csv(args.input)
    else:
        run_demo()
