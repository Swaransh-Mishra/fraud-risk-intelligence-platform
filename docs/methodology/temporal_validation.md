# Temporal Validation & Leakage Prevention

## 1. Objective

Fraud detection is a time-dependent problem. The model should be evaluated on transactions that occur after the transactions used for training.

The project therefore uses chronological validation rather than a random train/test split.

---

## 2. Dataset Timeline

The transaction dataset covers:

2018-04-01 to 2018-09-30

Total transactions:

1,754,155

Fraudulent transactions:

14,681

Fraud rate:

0.84%

---

## 3. Data Split

The dataset is divided chronologically into three periods.

| Period | Dates | Purpose |
|---|---|---|
| Training | 2018-04-01 to 2018-06-30 | Model development and training |
| Validation | 2018-07-01 to 2018-07-31 | Feature/model/threshold selection |
| Test | 2018-08-01 to 2018-09-30 | Final unseen evaluation |

The split is based on transaction timestamps rather than random sampling.

### Training Period

The training period is used for:

- feature development
- model training
- experiment tracking

### Validation Period

The validation period is used for:

- comparing feature sets
- model selection
- hyperparameter tuning
- decision-threshold selection

### Test Period

The test period represents future transactions that were not used during model development.

It is evaluated only after the final modeling decisions have been made.

## 4. Temporal Ordering

Transactions will be ordered by:

TX_DATETIME

Historical features must only use information available before the current transaction.

For transaction T:

Previous information
→ Feature calculation
→ Prediction for T

Future information must never be used to construct features for T.

---

## 5. Leakage Prevention Rules

### Customer Features

Customer historical statistics must be calculated using transactions occurring before the current transaction.

Examples:

- previous customer transaction count
- previous customer average amount
- previous customer maximum amount
- previous customer amount variability
- time since previous customer transaction

---

### Rolling Features

Rolling windows exclude the current transaction.

For a transaction occurring at time T:

### 1-hour window

Transactions in:

[T - 1 hour, T)

### 24-hour window

Transactions in:

[T - 24 hours, T)

The current transaction is excluded.

---

## 6. Terminal Features

Terminal historical statistics must also use only transactions that occurred before the current transaction.

Examples:

- previous terminal transaction count
- previous terminal average amount
- terminal transaction velocity

---

## 7. Target Leakage

`TX_FRAUD` is the prediction target.

It must never be used as an input feature.

`TX_FRAUD_SCENARIO` will not be used as a model feature because it directly describes the fraud scenario associated with the transaction.

It may be retained for post-model analysis and error analysis.

---

## 8. Model Evaluation Principle

Model development will follow:

Training
→ Validation
→ Final Test

The final test period will remain untouched during feature and model selection.

---

## 9. Feature Development Principle

Feature groups will be evaluated incrementally.

Baseline
→ Customer behavior
→ Rolling behavior
→ Terminal behavior
→ Behavioral deviation
→ Additional justified features

A feature will only be retained when it provides meaningful value without introducing leakage.

---

## 10. Reproducibility

The temporal split and feature-generation logic will be implemented in application code rather than manually performed inside the notebook.

The notebook will be used for analysis and validation of the pipeline.