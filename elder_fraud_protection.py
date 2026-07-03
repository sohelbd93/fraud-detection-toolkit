"""
elder_fraud_protection.py

Detection toolkit for financial exploitation patterns disproportionately
targeting older adults: sudden large withdrawals/transfers, newly added
payees receiving large sums shortly after being added, breaks from an
account's established spending routine (time-of-day, category, geography),
and caregiver/POA (power-of-attorney) abuse patterns (frequent
below-threshold transfers to a single new party).

This is a research/demo-grade reference implementation trained and
evaluated on synthetic data, intended for banks, credit unions, or
consumer-protection tooling to prototype against. It does not replace
adult-protective-services reporting workflows or bank compliance review.

Usage:
    python elder_fraud_protection.py --demo
    python elder_fraud_protection.py --input tx.csv
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# Rule-based red flags specific to elder financial exploitation
# ----------------------------------------------------------------------

NEW_PAYEE_WINDOW_DAYS = 14        # a payee added this recently is "new"
NEW_PAYEE_LARGE_AMOUNT = 2000     # large transfer to a brand-new payee
SUDDEN_WITHDRAWAL_MULTIPLIER = 4  # multiple of the account's normal transaction size
BELOW_THRESHOLD_REPEAT_COUNT = 5  # repeated transfers just under a reporting/review threshold
BELOW_THRESHOLD_AMOUNT = 3000
ROUTINE_HOUR_START, ROUTINE_HOUR_END = 7, 21  # typical active hours for account holder


@dataclass
class ElderFraudFlags:
    new_payee_large_transfer: bool
    sudden_large_withdrawal: bool
    off_hours_activity: bool
    repeated_below_threshold: bool

    def any_triggered(self) -> bool:
        return any([
            self.new_payee_large_transfer, self.sudden_large_withdrawal,
            self.off_hours_activity, self.repeated_below_threshold,
        ])


def flag_new_payee_large_transfer(df: pd.DataFrame) -> pd.Series:
    """First transfer to a payee, above a threshold, is a common financial
    exploitation opener (romance scams, fake tech-support, caregiver abuse)."""
    df_sorted = df.sort_values("timestamp")
    first_seen = df_sorted.groupby("payee_id")["timestamp"].transform("min")
    is_first_transfer = df_sorted["timestamp"] == first_seen
    is_recent_payee = (df_sorted["timestamp"] - first_seen).dt.days <= NEW_PAYEE_WINDOW_DAYS
    is_large = df_sorted["amount"] >= NEW_PAYEE_LARGE_AMOUNT
    flags = is_first_transfer & is_recent_payee & is_large
    return flags.reindex(df.index).fillna(False)


def flag_sudden_large_withdrawal(df: pd.DataFrame, account_col: str = "account_id") -> pd.Series:
    """Withdrawal far larger than the account's historical typical transaction."""
    flags = pd.Series(False, index=df.index)
    for acct, group in df.groupby(account_col):
        baseline = group["amount"].median()
        if baseline == 0:
            continue
        threshold = baseline * SUDDEN_WITHDRAWAL_MULTIPLIER
        flags.loc[group[group["amount"] >= threshold].index] = True
    return flags


def flag_off_hours_activity(df: pd.DataFrame) -> pd.Series:
    hour = df["timestamp"].dt.hour
    return (hour < ROUTINE_HOUR_START) | (hour > ROUTINE_HOUR_END)


def flag_repeated_below_threshold(df: pd.DataFrame, payee_col: str = "payee_id") -> pd.Series:
    """Multiple transfers to the same payee, each individually below a
    review/reporting threshold -- a pattern consistent with someone
    deliberately avoiding scrutiny (e.g. an abusive caregiver)."""
    flags = pd.Series(False, index=df.index)
    for payee, group in df.groupby(payee_col):
        below = group[group["amount"] < BELOW_THRESHOLD_AMOUNT]
        if len(below) >= BELOW_THRESHOLD_REPEAT_COUNT:
            flags.loc[below.index] = True
    return flags


def apply_rule_engine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["flag_new_payee_large_transfer"] = flag_new_payee_large_transfer(df)
    df["flag_sudden_large_withdrawal"] = flag_sudden_large_withdrawal(df)
    df["flag_off_hours_activity"] = flag_off_hours_activity(df)
    df["flag_repeated_below_threshold"] = flag_repeated_below_threshold(df)
    df["rule_flag_count"] = df[[
        "flag_new_payee_large_transfer", "flag_sudden_large_withdrawal",
        "flag_off_hours_activity", "flag_repeated_below_threshold",
    ]].sum(axis=1)
    return df


# ----------------------------------------------------------------------
# Unsupervised anomaly scoring -- deviation from an account's own routine
# ----------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"] = df["timestamp"].dt.hour
    acct_mean = df.groupby("account_id")["amount"].transform("mean")
    acct_std = df.groupby("account_id")["amount"].transform("std")
    df["amount_zscore_vs_account"] = (
        (df["amount"] - acct_mean) / acct_std.replace(0, np.nan)
    ).fillna(0)
    return df


def score_anomalies(df: pd.DataFrame, contamination: float = 0.03, seed: int = 42) -> pd.DataFrame:
    df = engineer_features(df)
    features = df[["amount", "rule_flag_count", "hour_of_day", "amount_zscore_vs_account"]].fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    model.fit(X)

    raw_scores = model.decision_function(X)
    df["anomaly_score"] = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min() + 1e-9)
    return df


def blended_risk_score(df: pd.DataFrame, rule_weight: float = 0.5, model_weight: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    max_flags = df["rule_flag_count"].max()
    normalized_rule_count = df["rule_flag_count"] / max_flags if max_flags > 0 else 0
    df["risk_score"] = rule_weight * normalized_rule_count + model_weight * df["anomaly_score"]
    df["review_candidate"] = df["risk_score"] >= df["risk_score"].quantile(0.97)
    return df


# ----------------------------------------------------------------------
# Synthetic demo data
# ----------------------------------------------------------------------

def generate_synthetic_transactions(n_accounts: int = 40, n_transactions: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    account_ids = [f"ACCT-{i:04d}" for i in range(n_accounts)]
    payee_pool = [f"PAYEE-{i:04d}" for i in range(200)]

    start = pd.Timestamp("2026-01-01")
    timestamps = start + pd.to_timedelta(rng.integers(0, 60 * 24 * 60, n_transactions), unit="m")

    df = pd.DataFrame({
        "transaction_id": [f"TX-{i:06d}" for i in range(n_transactions)],
        "account_id": rng.choice(account_ids, n_transactions),
        "payee_id": rng.choice(payee_pool, n_transactions),
        "timestamp": timestamps,
        "amount": np.round(rng.lognormal(mean=4.0, sigma=1.0, size=n_transactions), 2),
    })

    # inject a "new payee, large first transfer" exploitation pattern
    scam_idx = rng.choice(df.index, size=15, replace=False)
    scam_payee = "PAYEE-9999"
    df.loc[scam_idx, "payee_id"] = scam_payee
    df.loc[scam_idx, "amount"] = rng.uniform(2500, 8000, size=len(scam_idx))

    return df.sort_values("timestamp").reset_index(drop=True)


def run_demo() -> None:
    print("Generating synthetic transaction data...")
    df = generate_synthetic_transactions()

    print("Applying elder-fraud rule engine...")
    df = apply_rule_engine(df)

    print("Scoring with Isolation Forest...")
    df = score_anomalies(df)
    df = blended_risk_score(df)

    top = df.sort_values("risk_score", ascending=False).head(15)
    print(f"\nTotal transactions: {len(df)}")
    print(f"Review candidates (top ~3% risk score): {df['review_candidate'].sum()}")
    print("\nTop 15 highest-risk transactions:")
    print(top[["transaction_id", "account_id", "payee_id", "amount",
               "rule_flag_count", "anomaly_score", "risk_score", "review_candidate"]].to_string(index=False))


def score_csv(path: str) -> None:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"transaction_id", "account_id", "payee_id", "timestamp", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input CSV missing required columns: {missing}")

    df = apply_rule_engine(df)
    df = score_anomalies(df)
    df = blended_risk_score(df)

    out_path = path.replace(".csv", "_scored.csv")
    df.to_csv(out_path, index=False)
    print(f"Scored {len(df)} transactions -> {out_path}")
    print(f"Review candidates flagged: {df['review_candidate'].sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Run on generated synthetic data")
    parser.add_argument("--input", type=str, help="Path to a transaction CSV to score")
    args = parser.parse_args()

    if args.input:
        score_csv(args.input)
    else:
        run_demo()
