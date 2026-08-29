# 🛡️ ChargebackIQ — AI Risk Manager

**Track 02: Chargeback Evidence Responder**
*Stop the merchant losing money to fraud, returns and chargebacks*

🔗 **Live Demo:** [chargebackiq-aypyrdzpzmlxfpnnp7bvls.streamlit.app](https://chargebackiq-aypyrdzpzmlxfpnnp7bvls.streamlit.app)

## Problem

Merchants lose revenue to fraudulent and illegitimate chargebacks because manually reviewing disputes is slow and inconsistent. ChargebackIQ predicts the win-probability of a chargeback dispute from transaction evidence, auto-generates a fact-cited evidence packet for winnable cases, and routes uncertain cases to human review — cutting review time while keeping false-positive cost under control.

**Scope: Defense-only.** No autonomous financial actions. No evidence fabrication. This system evaluates and documents evidence; it takes no autonomous action and cannot be repurposed to fabricate disputes.

## Approach

1. **Evidence-based prediction**: A Logistic Regression model predicts dispute win-probability using verification signals (delivery confirmation, IP/AVS/CVV match), customer history, and dispute reason code.
2. **Cost-aware thresholding**: Instead of optimizing for accuracy alone, we modeled the real ₹ cost of false positives (auto-approving a losing dispute → lost amount + arbitration fee) vs. false negatives (missing a winnable dispute → unrecovered revenue), and swept thresholds to minimize total expected cost.
3. **Three-tier human-in-the-loop routing**: Auto-Approve (high confidence win) / Human Review (uncertain) / Auto-Reject (high confidence loss) — balancing automation with responsible oversight.
4. **Evidence packet generation**: For every case, a structured, fact-cited response is generated using only verified transaction fields — no fabricated claims.

## Results (Held-Out Test Set)

| Metric | Value |
|---|---|
| ROC-AUC | 0.791 |
| Precision / Recall (Won) | 0.77 / 0.89 |
| Precision / Recall (Lost) | 0.69 / 0.48 |
| Workload automated | 54.7% |
| Accuracy on auto-decided cases | 86.3% |
| Total cost @ default threshold | ₹375,020 |

All metrics reported on a held-out test set the model never saw during training or threshold tuning.

## Why Logistic Regression over XGBoost?

We evaluated both. XGBoost overfit on this dataset size (ROC-AUC 0.743 vs. 0.782 on validation) — we chose the simpler, more interpretable model because it generalized better, not because it's the default choice.

## Tech Stack

- Python, pandas, scikit-learn (Logistic Regression, StandardScaler)
- Streamlit (live demo UI)
- Matplotlib/Seaborn (EDA + cost-curve visualization)

## Project Structure

- `app.py` — Streamlit demo application
- `logreg_model.pkl`, `scaler.pkl`, `feature_cols.pkl` — trained model artifacts
- `requirements.txt` — dependencies

## Team / Author

Devansh
