# Feature Specification

## 1. Objective

The Fraud Risk Intelligence Platform uses engineered transaction, customer, and terminal behavioural features to identify potentially fraudulent transactions.

The feature pipeline is designed to preserve chronological transaction history and prevent the current transaction's own fraud outcome from directly influencing its model inputs.

The finalized model uses **23 approved features**.

---

## 2. Final Model Feature Schema

The model feature schema is centrally defined through the project's reusable preprocessing configuration.

The final 23 features are:

### Transaction and Temporal Features

| Feature | Description |
| --- | --- |
| `TX_AMOUNT` | Current transaction amount |
| `hour_of_day` | Hour in which the transaction occurred |
| `day_of_week` | Day of the week of the transaction |
| `is_weekend` | Indicator for weekend transactions |

### Customer Historical Behaviour

| Feature | Description |
| --- | --- |
| `customer_tx_count` | Historical number of transactions associated with the customer |
| `customer_avg_amount` | Historical average transaction amount for the customer |
| `customer_max_amount` | Historical maximum transaction amount for the customer |
| `customer_amount_std` | Historical standard deviation of the customer's transaction amounts |
| `time_since_customer_tx` | Time elapsed since the customer's previous transaction |

### Customer Amount Behaviour

| Feature | Description |
| --- | --- |
| `customer_amount_deviation` | Difference between the current transaction amount and the customer's historical spending behaviour |
| `customer_amount_ratio` | Ratio comparing the current transaction amount with the customer's historical spending level |

### Customer Recent Activity

| Feature | Description |
| --- | --- |
| `customer_tx_count_1h` | Customer transaction count within the recent one-hour window |
| `customer_tx_count_24h` | Customer transaction count within the recent 24-hour window |
| `customer_amount_sum_24h` | Customer transaction-amount total within the recent 24-hour window |

### Terminal Historical Behaviour

| Feature | Description |
| --- | --- |
| `terminal_tx_count` | Historical number of transactions associated with the terminal |
| `terminal_avg_amount` | Historical average transaction amount for the terminal |
| `terminal_max_amount` | Historical maximum transaction amount for the terminal |
| `terminal_amount_std` | Historical standard deviation of terminal transaction amounts |

### Terminal Fraud History

| Feature | Description |
| --- | --- |
| `terminal_fraud_count` | Historical number of fraud outcomes associated with the terminal |
| `terminal_fraud_rate` | Historical fraud rate associated with the terminal |

### Terminal Recent Activity

| Feature | Description |
| --- | --- |
| `terminal_tx_count_1h` | Terminal transaction count within the recent one-hour window |
| `terminal_tx_count_24h` | Terminal transaction count within the recent 24-hour window |
| `terminal_amount_sum_24h` | Terminal transaction-amount total within the recent 24-hour window |

---

## 3. Feature Availability

Feature generation is performed chronologically across the transaction history.

For behavioural features, information from transactions occurring before the current transaction can contribute to the current feature values.

This is particularly important for:

- customer transaction history
- customer amount statistics
- customer activity windows
- terminal transaction history
- terminal amount statistics
- terminal activity windows

The objective is to represent the information that would be available from transaction history when evaluating a subsequent transaction.

---

## 4. Historical Fraud Features

Two finalized features explicitly use historical fraud outcomes:

- `terminal_fraud_count`
- `terminal_fraud_rate`

These features are treated as sequential historical signals.

A previous transaction's fraud outcome may contribute to the terminal's historical fraud statistics for a subsequent transaction when that outcome is considered available to the scoring system.

This introduces an explicit operational assumption:

> Previous transaction outcomes are assumed to become available before subsequent transactions are scored.

The current transaction's own fraud label is not used to construct its terminal fraud-history features.

---

## 5. Current-Row Leakage Protection

The feature pipeline is designed so that the current transaction does not directly contribute its own target outcome to its model inputs.

Historical calculations are based on preceding transaction history, including prior observations used for customer and terminal behavioural statistics.

This distinction is important:

- **Current-row target leakage:** the transaction's own fraud label influences its features.
- **Historical-label availability:** previously observed fraud outcomes influence features for later transactions.

The first is treated as leakage and must be avoided.

The second is an explicit sequential-scoring assumption in this project.

---

## 6. Feature Groups

The final feature set can be viewed as five major signal groups:

```text
Transaction / Time
        ↓
Customer Behaviour
        ↓
Customer Recent Activity
        ↓
Terminal Behaviour
        ↓
Terminal Historical Fraud + Recent Activity