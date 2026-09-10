# Temporal Validation Methodology

## 1. Objective

Fraud transactions occur over time, and behavioural features depend on transaction history.

For this reason, the Fraud Risk Intelligence Platform uses a chronological evaluation strategy rather than a random train-test split.

The objective is to preserve the forward-looking structure of the fraud detection problem and prevent later transaction records from being placed into earlier model-development periods.

---

## 2. Dataset Timeline

The transaction dataset covers:

```text
April 2018 → September 2018