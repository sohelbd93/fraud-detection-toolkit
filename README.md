# Cyber Threat Sentinel

An open-source, real-time network threat detection system combining streaming
anomaly detection with periodic batch classification, benchmarked on public
intrusion-detection datasets (CICIDS2017 / UNSW-NB15).

> **What this is:** a research/demo-grade reference architecture for
> real-time ML-based network intrusion detection, released for the security
> and applied-ML community to learn from, benchmark against, and extend.
>
> **What this is not:** a production SOC replacement, and it is **not**
> trained or validated on real bank, payment-network, or utility traffic.
> All training/evaluation data is public benchmark data or synthetic traffic
> generated for demo purposes. Do not deploy against live infrastructure
> without your own security review, red-teaming, and compliance sign-off.

## Architecture

```
[Traffic Source]        synthetic generator, PCAP replay, or live capture
       |
       v
   [Kafka topic: raw-flows]
       |
       v
[Streaming feature pipeline]   Faust/Flink: windowing, flow feature extraction
       |
       v
[Online detection model]        river (Half-Space Trees / Adaptive Random Forest)
       |                        scores every flow in near real time
       v
   [Kafka topic: alerts] --------------------------+
       |                                            |
       v                                            v
[Batch re-scoring model]                    [FastAPI alert feed]
  XGBoost / PyTorch autoencoder,                    |
  trained offline on CICIDS2017,                    v
  periodically re-scores flagged                [Dashboard]
  events for higher precision
       |
       v
[Model registry + evaluation reports]
```

Two-tier design rationale: a pure streaming model gets you speed but weak
precision; a pure batch model gets you accuracy but no real-time claim.
Combining them mirrors how production tooling (e.g. Zeek + ML scoring) is
typically structured, while staying honest about each tier's tradeoffs.

## Datasets

- **CICIDS2017 / CSE-CIC-IDS2018** — primary benchmark, labeled flow-level
  CSVs (CICFlowMeter features) plus raw PCAP.
- **UNSW-NB15** — secondary validation set with a different feature
  extraction methodology, used to check the model isn't just overfitting to
  CICIDS's specific artifacts.
- **Synthetic generator** (`ingestion/synthetic_flow_generator.py`) — for
  quick local demos without downloading the full datasets.

Datasets are not vendored in this repo (large + licensing). See
`data/README.md` for download instructions.

## Repo layout

| Path | Purpose |
|---|---|
| `ingestion/` | Kafka producers: synthetic flow generator, PCAP replay, live capture |
| `streaming/` | Feature windowing + online model scoring job |
| `models/online/` | River-based incremental anomaly/classification models |
| `models/batch/` | PyTorch autoencoder + XGBoost classifier, training scripts |
| `models/registry/` | Versioned model artifacts, MLflow tracking config |
| `detection_engine/` | Scoring orchestration, thresholding, alert generation |
| `api/` | FastAPI service: alert feed, model metrics, health |
| `dashboard/` | Lightweight Streamlit dashboard for live alerts |
| `evaluation/` | Benchmark scripts + metrics reports against CICIDS/UNSW-NB15 |
| `docs/` | Architecture notes, threat model, dataset notes |

## Quickstart

```bash
docker-compose up -d          # Kafka + Zookeeper + API + dashboard
pip install -r requirements.txt
python ingestion/synthetic_flow_generator.py     # start producing demo traffic
python streaming/online_scorer.py                # start online scoring
uvicorn api.main:app --reload                    # alert API on :8000
streamlit run dashboard/app.py                   # live dashboard on :8501
```

## Roadmap

- [x] Repo scaffold, synthetic data generator, online model stub
- [ ] Kafka producer for CICIDS2017 flow replay
- [ ] Faust streaming feature pipeline
- [ ] River online model wired end-to-end
- [ ] XGBoost/autoencoder batch classifier + evaluation report
- [ ] FastAPI alert feed
- [ ] Streamlit dashboard
- [ ] Benchmark results published in `evaluation/results.md`

## License

MIT (see `LICENSE`). Contributions welcome — see `CONTRIBUTING.md`.

## Responsible use

This project is for research, education, and defensive security tooling.
It must not be used to build offensive tooling, evade detection systems, or
target systems you don't own or have authorization to test. See
`SECURITY.md` for the disclosure policy.
