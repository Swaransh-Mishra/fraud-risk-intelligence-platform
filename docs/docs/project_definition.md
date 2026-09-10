# Fraud Risk Intelligence Platform

## Project Definition

### 1. Business Problem

Financial transaction systems need to identify potentially fraudulent activity while limiting unnecessary intervention on legitimate customers.

Fraud detection is therefore not only a classification problem. A practical fraud-risk system must generate useful fraud probabilities, convert those probabilities into operational decisions, evaluate the cost of different decision thresholds, expose the finalized model through an API, and provide a foundation for monitoring and future model improvement.

The **Fraud Risk Intelligence Platform** is an end-to-end machine learning system designed to demonstrate this workflow for transaction fraud detection.

---

## 2. Objective

The primary objective is to build a production-oriented fraud-risk platform that:

- predicts the probability that a transaction is fraudulent
- converts fraud probabilities into configurable risk decisions
- uses customer and terminal behavioural history as predictive signals
- evaluates models using fraud-focused ranking and classification metrics
- analyzes threshold and false-positive / false-negative trade-offs
- evaluates business cost under explicit analytical assumptions
- exposes the finalized model through an API
- provides a Streamlit interface for interactive prediction and monitoring
- records prediction activity for operational analysis
- provides a foundation for model monitoring and future retraining workflows

The project emphasizes the complete machine learning lifecycle rather than model training alone.

---

## 3. Target Users

### Fraud Analysts

Review transactions with elevated fraud probability and prioritize suspicious activity for investigation.

### Risk Operations Teams

Use fraud probabilities, risk levels, and decision thresholds to support transaction-review workflows.

### ML Engineers

Work with the modeling pipeline, reusable application components, model artifacts, evaluation results, and monitoring foundation.

---

## 4. Data and Modeling Scope

The project uses a transaction dataset containing approximately **1.75 million transactions** covering **April through September 2018**.

The fraud class represents a small minority of transactions, making class imbalance an important consideration during model development and evaluation.

The finalized model uses **23 engineered features** covering:

- transaction and temporal behaviour
- customer historical behaviour
- customer rolling activity
- customer transaction-amount deviation
- terminal historical behaviour
- terminal fraud history
- terminal rolling activity

The final feature matrix is shared consistently across the training, validation, and test datasets.

---

## 5. Chronological Evaluation Design

The project uses chronological evaluation rather than a random train-test split.

| Dataset | Period | Purpose |
| --- | --- | --- |
| Training | April–July 2018 | Model fitting |
| Validation | August 2018 | Model comparison, threshold analysis, tuning, business-cost analysis, and model selection |
| Test | September 2018 | Final evaluation after the production configuration is frozen |

The chronological structure is implemented through the reusable project splitting logic.

The test set is not used to select the final model, ensemble weighting, or production threshold.

---

## 6. Feature Engineering Approach

The feature pipeline generates behavioural signals from transaction history.

Customer-level features capture patterns such as:

- transaction frequency
- historical average and maximum transaction amount
- transaction-amount deviation
- transaction-amount ratio
- recent transaction activity

Terminal-level features capture patterns such as:

- historical transaction activity
- historical transaction amounts
- recent terminal activity
- historical fraud count
- historical fraud rate

Historical features are generated chronologically so that the current transaction's own outcome is not directly used to construct its features.

Terminal fraud-history features require an explicit sequential-scoring assumption: previously observed transaction outcomes are treated as becoming available for subsequent transactions.

---

## 7. Modeling Strategy

The modeling workflow begins with interpretable and non-linear reference models before evaluating stronger tree-based candidates.

The final production configuration uses a probability-weighted ensemble of two randomized-search finalists:

**RandomizedSearch XGBoost + RandomizedSearch CatBoost**

with:

- **XGBoost weight:** 0.3
- **CatBoost weight:** 0.7
- **Production decision threshold:** 0.70

The ensemble combines positive-class fraud probabilities rather than combining hard class predictions.

The final configuration was selected using validation-stage predictive performance, threshold behaviour, business-cost analysis, and robustness review.

---

## 8. Evaluation Philosophy

Because fraud is a highly imbalanced classification problem, accuracy is not treated as the primary model-selection metric.

The evaluation considers:

- PR-AUC
- ROC-AUC
- precision
- recall
- F1-score
- false positives
- false negatives
- true positives
- true negatives
- business cost across thresholds
- threshold sensitivity and robustness

PR-AUC receives particular attention because it evaluates precision-recall behaviour in a setting where fraudulent transactions are rare.

---

## 9. Production Decision Strategy

Model probabilities are converted into fraud decisions using a decision threshold.

The threshold is treated as an operational parameter rather than an intrinsic property of the model.

Validation-stage analysis examines how different thresholds affect:

- fraud detection
- investigation workload
- false negatives
- precision
- recall
- F1-score
- estimated business cost

The final production threshold is selected during validation and remains fixed during final test evaluation.

---

## 10. Business-Cost Framework

The project uses explicit analytical cost assumptions to compare operating points:

- **False Positive cost = 5**
- **False Negative cost = 100**

The higher false-negative cost represents the assumption that missing fraudulent activity is substantially more costly than reviewing a legitimate transaction.

These values are analytical assumptions for model evaluation and are not presented as real-world company costs.

Business cost is calculated as:

```text
Total Cost =
(False Positives × 5) +
(False Negatives × 100)