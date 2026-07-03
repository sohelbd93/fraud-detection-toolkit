# Fraud Detection Toolkit

Two standalone financial-fraud detection scripts combining rule-based red
flags with unsupervised anomaly scoring (Isolation Forest), plus a blended
risk score for analyst triage.

> **Scope note:** these are research/demo-grade reference implementations,
> trained and evaluated on synthetic data generated in-script. They are not
> certified AML/compliance systems and do not replace a bank's BSA/AML
> program or adult-protective-services reporting workflows. Do not point
> these at real customer data without your own compliance, privacy, and
> security review.

## Scripts

### `aml_detection.py` — Anti-Money Laundering transaction monitoring
Detects classic AML red flags and blends them with an anomaly model:
- **Structuring/smurfing** — deposits just under the $10,000 CTR threshold
- **Round-number transactions** — suspiciously round amounts
- **High velocity** — accounts with abnormally many transactions in a
  rolling 24h window
- **Rapid movement / layering** — funds received and moved out again to a
  different counterparty within 72h
- Outputs a blended `risk_score` and flags the top ~3% as `sar_candidate`
  (Suspicious Activity Report candidates)

### `elder_fraud_protection.py` — Elder financial exploitation detection
Detects patterns disproportionately associated with financial abuse of
older adults:
- **New payee, large transfer** — first-ever transfer to a payee, above a
  threshold, within days of that payee first appearing (common opener for
  romance scams, tech-support scams, caregiver abuse)
- **Sudden large withdrawal** — far above the account's own historical
  transaction size
- **Off-hours activity** — transactions outside the account's typical
  active hours
- **Repeated below-threshold transfers** — many transfers to the same
  payee, each individually below a review threshold (pattern consistent
  with deliberately avoiding scrutiny)
- Outputs a blended `risk_score` and flags the top ~3% as `review_candidate`

## Usage

```bash
pip install -r requirements.txt

# Run on generated synthetic data
python aml_detection.py --demo
python elder_fraud_protection.py --demo

# Score your own transaction CSV
python aml_detection.py --input transactions.csv
python elder_fraud_protection.py --input transactions.csv
```

**Required CSV columns:**
- `aml_detection.py`: `transaction_id, account_id, counterparty_id, timestamp, direction, amount`
- `elder_fraud_protection.py`: `transaction_id, account_id, payee_id, timestamp, amount`

Output is written alongside the input as `<input>_scored.csv`.

## How the risk score works

Each script runs a rule engine first (fast, explainable, auditable — the
kind of logic a compliance team can review line by line), then trains an
Isolation Forest on engineered features (transaction amount, per-account
z-score, hour of day, rule-flag count) to catch anomalies the fixed rules
miss. The two signals are blended into a single `risk_score` so results can
be sorted for analyst review rather than treated as a hard yes/no.

## License

MIT — see `LICENSE`.

## Responsible use

Intended for fraud-research, education, and prototyping by banks, credit
unions, and consumer-protection tooling teams. Do not use to test against
real customer accounts without authorization, and do not use for building
tools that facilitate evading fraud detection.
