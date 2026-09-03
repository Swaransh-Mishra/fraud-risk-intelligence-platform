# Feature Engineering Specification

## 1. Objective

The feature-engineering pipeline converts raw transaction data into leakage-safe behavioral features for fraud-risk prediction.

Features must use only information that would have been available at the time of the transaction being scored.

The raw dataset will not be modified.

---

## 2. Feature Groups

The feature pipeline will contain the following groups:

1. Transaction and temporal features
2. Customer historical behavior
3. Customer rolling behavior
4. Customer behavioral deviation
5. Terminal historical behavior
6. Terminal rolling behavior
7. Cross-behavior features

---

## 3. Transaction and Temporal Features

| Feature | Source | Description |
|---|---|---|
| `transaction_amount` | `TX_AMOUNT` | Current transaction amount |
| `hour_of_day` | `TX_DATETIME` | Hour in which transaction occurred |
| `day_of_week` | `TX_DATETIME` | Day of week |
| `is_weekend` | `TX_DATETIME` | Weekend indicator |

`TX_DATETIME` will be used to derive temporal features rather than being passed directly to the model.

---

## 4. Customer Historical Features

Customer history will be calculated using only transactions occurring before the current transaction.

| Feature | Description |
|---|---|
| `customer_tx_count` | Number of previous customer transactions |
| `customer_avg_amount` | Historical average transaction amount |
| `customer_max_amount` | Historical maximum transaction amount |
| `customer_amount_std` | Historical transaction amount variability |
| `time_since_customer_tx` | Time since the customer's previous transaction |

The current transaction must not contribute to these statistics.

---

## 5. Customer Rolling Features

Rolling windows will measure recent customer activity.

### 1-hour window

| Feature | Description |
|---|---|
| `customer_tx_count_1h` | Customer transactions during the previous hour |
| `customer_amount_sum_1h` | Customer transaction value during the previous hour |

### 24-hour window

| Feature | Description |
|---|---|
| `customer_tx_count_24h` | Customer transactions during the previous 24 hours |
| `customer_amount_sum_24h` | Customer transaction value during the previous 24 hours |

The current transaction is excluded from all rolling windows.

---

## 6. Customer Behavioral Deviation

The model should identify transactions that are unusual relative to the customer's historical behavior.

| Feature | Description |
|---|---|
| `customer_amount_ratio` | Current amount relative to previous customer average |
| `customer_amount_deviation` | Difference between current amount and previous customer behavior |

The historical reference must only contain information available before the current transaction.

---

## 7. Terminal Historical Features

Terminal behavior will be represented using historical transactions occurring before the current transaction.

| Feature | Description |
|---|---|
| `terminal_tx_count` | Previous transactions at the terminal |
| `terminal_avg_amount` | Historical average transaction amount at the terminal |

---

## 8. Terminal Rolling Features

### 1-hour window

| Feature | Description |
|---|---|
| `terminal_tx_count_1h` | Terminal transactions during the previous hour |
| `terminal_amount_sum_1h` | Terminal transaction value during the previous hour |

### 24-hour window

| Feature | Description |
|---|---|
| `terminal_tx_count_24h` | Terminal transactions during the previous 24 hours |
| `terminal_amount_sum_24h` | Terminal transaction value during the previous 24 hours |

The current transaction is excluded.

---

## 9. Identifier Handling

`CUSTOMER_ID` and `TERMINAL_ID` will not be directly used as high-cardinality categorical model features.

Instead, they will be used to construct behavioral features.

---

## 10. Excluded Features

### `TX_FRAUD`

This is the prediction target and must never be used as a model feature.

### `TX_FRAUD_SCENARIO`

This field directly describes the fraud scenario and will not be used as a model input.

It may be retained for post-model analysis and error analysis.

### Raw identifiers

`CUSTOMER_ID` and `TERMINAL_ID` will not be directly passed to the baseline model.

---

## 11. Leakage Prevention

For every transaction at time T:

- historical features use information before T
- rolling windows end immediately before T
- the current transaction is excluded
- future transactions are never used
- target labels are never used to construct ordinary behavioral features

The feature pipeline must preserve chronological ordering.

---

## 12. Feature Development Strategy

Features will be introduced incrementally.

### Feature Set 1 — Baseline

- `transaction_amount`
- `hour_of_day`
- `day_of_week`
- `is_weekend`

### Feature Set 2 — Customer Behavior

Add customer historical features.

### Feature Set 3 — Customer Rolling Behavior

Add 1-hour and 24-hour customer activity features.

### Feature Set 4 — Terminal Behavior

Add historical and rolling terminal activity.

### Feature Set 5 — Behavioral Deviation

Add customer amount deviation and ratio features.

Additional features will only be retained when they provide measurable value without introducing leakage.

---

## 13. Validation Requirements

The feature pipeline must be validated for:

- row-count preservation
- missing values
- infinite values
- correct data types
- chronological ordering
- leakage
- feature distributions
- reproducibility

Feature engineering will be implemented in application code rather than manually inside the notebook.