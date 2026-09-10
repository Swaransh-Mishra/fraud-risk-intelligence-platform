# Fraud Risk Intelligence Platform

> An end-to-end machine learning and MLOps-oriented platform for transaction fraud detection, risk scoring, model evaluation, API serving, and operational monitoring.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interface-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-82%20passed-success)](#automated-testing)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#license)

---

## Table of Contents

- [Fraud Risk Intelligence Platform](#fraud-risk-intelligence-platform)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Business Problem](#business-problem)
  - [Project Objective](#project-objective)
  - [Why This Project](#why-this-project)
  - [Data \& Feature Engineering](#data--feature-engineering)
    - [Dataset](#dataset)
  - [Feature Engineering](#feature-engineering)
    - [Feature Groups](#feature-groups)
      - [Transaction \& Temporal Signals](#transaction--temporal-signals)
      - [Customer Behaviour](#customer-behaviour)
      - [Customer Amount Behaviour](#customer-amount-behaviour)
      - [Customer Recent Activity](#customer-recent-activity)
      - [Terminal Behaviour](#terminal-behaviour)
      - [Terminal Fraud History](#terminal-fraud-history)
      - [Terminal Recent Activity](#terminal-recent-activity)
  - [Chronological Data Splitting](#chronological-data-splitting)
  - [Modeling Strategy](#modeling-strategy)
  - [Model Development Workflow](#model-development-workflow)
  - [Final Model Results](#final-model-results)
  - [Final Champion](#final-champion)
  - [Validation vs Final Test](#validation-vs-final-test)
  - [Final Test Confusion Matrix](#final-test-confusion-matrix)
  - [Production System \& MLOps](#production-system--mlops)
  - [Production Architecture](#production-architecture)
  - [Repository Structure](#repository-structure)
  - [Visual Showcase](#visual-showcase)
    - [Application Interface](#application-interface)
      - [Platform Overview](#platform-overview)
      - [Fraud Prediction](#fraud-prediction)
      - [Batch Prediction](#batch-prediction)
      - [Monitoring](#monitoring)
  - [Model Evaluation Visualizations](#model-evaluation-visualizations)
    - [Model Comparison](#model-comparison)
    - [Threshold Trade-off](#threshold-trade-off)
    - [Business Cost Analysis](#business-cost-analysis)
  - [Technical Stack](#technical-stack)
  - [Engineering \& ML Practices](#engineering--ml-practices)
    - [Chronological Evaluation](#chronological-evaluation)
    - [Leakage-Aware Feature Engineering](#leakage-aware-feature-engineering)
    - [Validation-Driven Model Selection](#validation-driven-model-selection)
    - [Frozen Final Holdout](#frozen-final-holdout)
    - [Reusable Application Components](#reusable-application-components)
    - [Metadata-Driven Inference](#metadata-driven-inference)
    - [Automated Testing](#automated-testing)
    - [Containerized Serving](#containerized-serving)
  - [Assumptions \& Limitations](#assumptions--limitations)
    - [Sequential Historical Information](#sequential-historical-information)
    - [Analytical Business Costs](#analytical-business-costs)
  - [Deployment](#deployment)
  - [Running Locally](#running-locally)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
    - [3. Install Dependencies](#3-install-dependencies)
    - [4. Start the FastAPI Backend](#4-start-the-fastapi-backend)
    - [5. Verify the Backend](#5-verify-the-backend)
    - [6. Explore the API with Swagger](#6-explore-the-api-with-swagger)
    - [7. Start the Streamlit Frontend](#7-start-the-streamlit-frontend)
    - [8. Use the Platform](#8-use-the-platform)
    - [9. Run the Test Suite](#9-run-the-test-suite)
    - [Local Runtime Architecture](#local-runtime-architecture)
    - [Complete Local Startup](#complete-local-startup)
    - [Development Notes](#development-notes)
  - [Swaransh Mishra](#swaransh-mishra)
    - [Connect with Me](#connect-with-me)
  - [License](#license)

---

## Overview

The **Fraud Risk Intelligence Platform** is an end-to-end machine learning system designed to identify potentially fraudulent financial transactions and convert model predictions into operational risk decisions.

The project goes beyond model training by implementing the broader machine learning lifecycle:

**Data → Feature Engineering → Model Development → Validation → Threshold Selection → Business Cost Analysis → Model Serving → Monitoring**

The platform combines behavioural transaction features, customer history, terminal history, model evaluation, probability-based ensemble prediction, API serving, interactive visualization, prediction logging, drift detection, and automated testing into a single portfolio-scale system.

The project is designed with a **machine learning + MLOps-oriented architecture**, separating model-development workflows from reusable application components used for inference and monitoring.

---

## Business Problem

Fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a small proportion of total transaction activity.

A useful fraud-risk system therefore needs to do more than maximize accuracy.

It should be able to:

- identify suspicious transactions
- estimate fraud probability
- support configurable decision thresholds
- balance false positives against false negatives
- evaluate business-cost trade-offs
- preserve chronological evaluation
- expose predictions through an API
- support batch transaction scoring
- record prediction activity
- monitor model behaviour and data drift

The platform addresses these requirements through a complete machine learning workflow built around transaction-level behavioural signals.

---

## Project Objective

The primary objective is to build a realistic fraud-risk platform that demonstrates how a machine learning model can move from historical transaction data to an operational prediction service.

The system focuses on:

- **Fraud probability prediction**
- **Behavioural feature engineering**
- **Chronological model validation**
- **Imbalanced classification evaluation**
- **Threshold selection**
- **Business-cost analysis**
- **Probability-weighted ensemble modeling**
- **Model artifact and metadata management**
- **FastAPI model serving**
- **Streamlit-based interaction**
- **Prediction logging**
- **Data drift detection**
- **Performance monitoring**
- **Automated testing**
- **Dockerized API deployment**

---

## Why This Project

The project is intentionally structured as more than a standalone machine learning notebook.

The objective is to demonstrate the complete path from:

```text
Historical Transactions
        ↓
Data & Feature Engineering
        ↓
Model Development
        ↓
Validation & Model Selection
        ↓
Decision Threshold
        ↓
Business Evaluation
        ↓
Persisted Model
        ↓
API Serving
        ↓
Interactive Application
        ↓
Logging & Monitoring
```
## Data & Feature Engineering

### Dataset

The platform is developed using a transaction-level fraud dataset containing approximately **1.75 million transactions** across a six-month period.

| Attribute | Value |
| --- | ---: |
| Total Transactions | 1,754,155 |
| Fraudulent Transactions | 14,681 |
| Fraud Prevalence | ~0.84% |
| Customers | 4,990 |
| Terminals | 10,000 |
| Time Period | April–September 2018 |
| Raw Columns | 9 |

The low fraud prevalence creates a highly imbalanced classification problem, where a model can achieve high overall accuracy while still performing poorly on the minority fraud class.

For this reason, the project emphasizes **precision-recall behaviour, PR-AUC, F1-score, recall, false positives, false negatives, and business cost** rather than relying on accuracy alone.

---

## Feature Engineering

The final model uses **23 model features** combining transaction-level information with historical customer and terminal behaviour.

### Feature Groups

#### Transaction & Temporal Signals

- `TX_AMOUNT`
- `hour_of_day`
- `day_of_week`
- `is_weekend`

These features capture the characteristics and timing of the current transaction.

#### Customer Behaviour

- `customer_tx_count`
- `customer_avg_amount`
- `customer_max_amount`
- `customer_amount_std`
- `time_since_customer_tx`

These features describe the customer's historical transaction behaviour.

#### Customer Amount Behaviour

- `customer_amount_deviation`
- `customer_amount_ratio`

These features compare the current transaction against the customer's historical spending pattern.

#### Customer Recent Activity

- `customer_tx_count_1h`
- `customer_tx_count_24h`
- `customer_amount_sum_24h`

These features capture short-term transaction frequency and spending activity.

#### Terminal Behaviour

- `terminal_tx_count`
- `terminal_avg_amount`
- `terminal_max_amount`
- `terminal_amount_std`

These features represent the historical behaviour of the terminal associated with the transaction.

#### Terminal Fraud History

- `terminal_fraud_count`
- `terminal_fraud_rate`

These features capture historical fraud activity associated with the terminal.

#### Terminal Recent Activity

- `terminal_tx_count_1h`
- `terminal_tx_count_24h`
- `terminal_amount_sum_24h`

These features capture recent transaction activity at the terminal level.

---

## Chronological Data Splitting

Fraud detection is inherently time-dependent because behavioural features are constructed from transaction history.

Instead of randomly splitting the dataset, the project uses a **chronological train / validation / test strategy**.

```text
April ───────── July │ August │ September
        Training     │  Valid.│   Test
                     │        │
              Model Development
                              │
                         Final Holdout
```
## Modeling Strategy

The modeling workflow evaluates fraud detection models progressively, starting with a baseline and moving toward stronger non-linear models and tuned ensemble configurations.

The objective is not simply to maximize accuracy, but to identify a model that provides strong fraud ranking and classification performance under severe class imbalance while remaining suitable for operational decision-making.

---

## Model Development Workflow

```text
Logistic Regression Baseline
          ↓
Tree-Based Candidate Models
          ↓
Model Comparison
          ↓
Hyperparameter Tuning
          ↓
Tuned XGBoost + Tuned CatBoost
          ↓
Probability-Weighted Ensembles
          ↓
Threshold Analysis
          ↓
Business-Cost & Robustness Analysis
          ↓
Final Champion
```
## Final Model Results

After the model, ensemble configuration, and production threshold were selected using the validation period, the final configuration was frozen and evaluated on the **September 2018 test set**.

The test set was not used during model selection or threshold optimization.

---

## Final Champion

**RandomizedSearch XGBoost + RandomizedSearch CatBoost**

| Configuration | Value |
| --- | --- |
| XGBoost Weight | 0.3 |
| CatBoost Weight | 0.7 |
| Production Threshold | 0.70 |
| Model Type | Weighted Probability Ensemble |
| Model Features | 23 |

The ensemble produces a continuous fraud probability and applies the frozen production threshold to generate the final fraud decision.

---

## Validation vs Final Test

The difference between validation and final test performance provides an indication of how the selected configuration generalizes to a later time period.

| Metric | Validation | September Test |
| --- | ---: | ---: |
| PR-AUC | 0.746233 | 0.584143 |
| ROC-AUC | — | 0.972573 |
| Precision | 0.729565 | 0.473150 |
| Recall | 0.672162 | 0.622693 |
| F1-score | 0.699688 | 0.537718 |

The September test results are lower than the validation results, particularly for precision and PR-AUC.

This can occur in a time-based fraud detection problem because the test period represents a later transaction distribution that was not used for model selection.

The final test results should therefore be interpreted as an **unseen temporal holdout evaluation**, not as a continuation of validation tuning.

---

## Final Test Confusion Matrix

At the frozen production threshold of **0.70**, the final September test results were:

| Outcome | Count |
| --- | ---: |
| True Negatives | 283,560 |
| False Positives | 1,766 |
| False Negatives | 961 |
| True Positives | 1,586 |

The model correctly identifies a substantial portion of fraudulent transactions while limiting the number of legitimate transactions classified as fraud.

```text
                    Predicted
                 Non-Fraud    Fraud
              ┌────────────┬──────────┐
Actual Fraud  │    961     │   1,586  │
              │    FN      │    TP    │
              ├────────────┼──────────┤
Actual Legit. │  283,560   │   1,766  │
              │    TN      │    FP    │
              └────────────┴──────────┘
```
## Production System & MLOps

The project is structured as a production-oriented machine learning application rather than a notebook-only model.

The system separates **model development**, **model serving**, **application interaction**, and **operational monitoring** into reusable components.

---

## Production Architecture

The platform separates model development from reusable inference, application, and monitoring components.

![Production Architecture](assets/plots/production_architecture.png)


## Repository Structure

```text
Fraud-Risk-Intelligence-Platform/
│
├── app/
│   ├── api/
│   │   └── schemas.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   ├── handlers.py
│   │   ├── logging.py
│   │   └── ...
│   │
│   ├── data_loader/
│   │   ├── loader.py
│   │   └── split.py
│   │
│   ├── evaluation/
│   │   ├── business.py
│   │   ├── error_analysis.py
│   │   ├── explainability.py
│   │   ├── metrics.py
│   │   ├── model_comparison.py
│   │   └── thresholds.py
│   │
│   ├── features/
│   │   ├── customer.py
│   │   ├── pipeline.py
│   │   ├── preprocessing.py
│   │   ├── temporal.py
│   │   └── terminal.py
│   │
│   ├── inference/
│   │   └── predictor.py
│   │
│   ├── models/
│   │   ├── advanced.py
│   │   ├── baseline.py
│   │   ├── ensemble.py
│   │   ├── final_model.py
│   │   ├── model_io.py
│   │   └── tuning.py
│   │
│   ├── monitoring/
│   │   ├── analytics.py
│   │   ├── drift.py
│   │   ├── logging.py
│   │   └── performance.py
│   │
│   └── tracking/
│       └── experiment.py
│
├── artifacts/
│   ├── fraud_risk_model.joblib
│   └── model_metadata.json
│
├── assets/
│   └── plots/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── docs/
│   ├── docs/
│   │   └── project_definition.md
│   └── methodology/
│       ├── feature_specification.md
│       └── temporal_validation.md
│
├── notebooks/
│   ├── 01_data_and_feature_analysis.ipynb
│   └── 02_model_development_and_evaluation.ipynb
│
├── scripts/
│   └── create_sample_batch.py
│
├── streamlit/
│   └── app.py
│
├── tests/
│   ├── test_api.py
│   ├── test_drift.py
│   ├── test_monitoring.py
│   ├── test_performance.py
│   ├── test_predictor.py
│   └── ...
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```
## Visual Showcase

The platform includes visual artifacts covering the modeling workflow, final evaluation, explainability, and deployed application interface.

---

### Application Interface

#### Platform Overview

The Streamlit overview provides a high-level view of the platform health, active model configuration, and prediction activity.

![Platform Overview](assets/plots/streamlit_overview.png)

---

#### Fraud Prediction

The fraud-prediction interface allows a transaction to be scored using the finalized 23-feature model schema.

![Fraud Prediction](assets/plots/streamlit_fraud_prediction.png)

---

#### Batch Prediction

The batch-prediction interface supports CSV-based transaction scoring and presents the resulting fraud probabilities and decisions.

![Batch Prediction](assets/plots/streamlit_batch_prediction.png)

---

#### Monitoring

The monitoring interface provides visibility into prediction activity and available operational monitoring information.

![Monitoring](assets/plots/streamlit_monitoring.png)

---

## Model Evaluation Visualizations

### Model Comparison

Validation PR-AUC is used to compare the candidate model configurations under the highly imbalanced fraud classification setting.

![Model Comparison](assets/plots/model_comparison_pr_auc.png)

---

### Threshold Trade-off

The threshold analysis shows how the operating point affects predictive behaviour across different decision thresholds.

![Threshold Trade-off](assets/plots/final_champion_threshold_tradeoff.png)

The production threshold selected through the validation workflow is **0.70**.

---

### Business Cost Analysis

The business-cost analysis evaluates the trade-off between false-positive and false-negative costs across possible operating thresholds.

![Business Cost Analysis](assets/plots/business_cost_analysis.png)

The analysis uses the project's explicit analytical assumptions:

```text
False Positive Cost = 5
False Negative Cost = 100
```
## Technical Stack

| Layer | Technologies |
| --- | --- |
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy, PyArrow |
| Machine Learning | Scikit-learn, XGBoost, CatBoost, LightGBM |
| Statistical / Scientific Computing | SciPy |
| Model Persistence | Joblib |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| Visualization | Matplotlib |
| Testing | Pytest, HTTPX |
| Notebook Environment | Jupyter, IPython Kernel |
| Containerization | Docker |
| Experiment Tracking | File-based project tracking |

---

## Engineering & ML Practices

The project follows several practices intended to keep the system maintainable and evaluation results meaningful.

### Chronological Evaluation

Training, validation, and test data are separated according to transaction time rather than through a random split.

### Leakage-Aware Feature Engineering

Historical customer and terminal features are generated using preceding transaction history.

### Validation-Driven Model Selection

Model, ensemble, and threshold decisions are made using the validation period.

### Frozen Final Holdout

The September 2018 test period is evaluated only after the final model configuration and production threshold are frozen.

### Reusable Application Components

Core inference, monitoring, evaluation, feature, and model functionality is implemented outside the notebooks.

### Metadata-Driven Inference

The production inference layer loads the persisted model configuration and decision threshold from model metadata.

### Automated Testing

The application is covered by an automated test suite with **82 passing tests** in the latest verified run.

### Containerized Serving

The FastAPI prediction service can be packaged and executed through Docker.

---

## Assumptions & Limitations

The platform is designed as a portfolio-scale machine learning and MLOps-oriented system. It demonstrates production-oriented practices without claiming enterprise-scale infrastructure.

### Sequential Historical Information

The terminal fraud-history features assume that previously processed fraud outcomes become available before subsequent transactions are scored.

This assumption should be considered when adapting the feature pipeline to a real-time production environment.

### Analytical Business Costs

The business-cost analysis uses assumed costs:

```text
False Positive = 5
False Negative = 100
```
The business-cost analysis is used as supporting decision evidence; the frozen production threshold of **0.70** was selected through the validation-based model evaluation workflow.

## Deployment

The platform is designed to be deployed as separate application services:

```text
                    Cloud Deployment
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
       FastAPI Backend           Streamlit Frontend
             │                         │
             └────────────┬────────────┘
                          ▼
                  Fraud Risk Platform
```
## Running Locally

The Fraud Risk Intelligence Platform can be run locally using a Python virtual environment. The platform uses **FastAPI** as the backend inference service and **Streamlit** as the interactive frontend.

The finalized model artifact and metadata are loaded at runtime, so model training is not required to run the application.

### 1. Clone the Repository

Clone the repository and move into the project directory.

```bash
git clone https://github.com/Swaransh-Mishra/Fraud-Risk-Intelligence-Platform.git
cd Fraud-Risk-Intelligence-Platform
```
### 2. Create a Virtual Environment

Create an isolated Python environment to keep project dependencies separate from the system Python installation.

```bash
python -m venv .venv
```
Activate the environment based on your operating system.

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```
**macOS / Linux**

```bash
source .venv/bin/activate
```
### 3. Install Dependencies

Install the required project dependencies from `requirements.txt`.

```bash
pip install -r requirements.txt
```
### 4. Start the FastAPI Backend

FastAPI provides the reusable inference API for fraud prediction, batch prediction, model information, and monitoring.

Start the backend with:

```bash
uvicorn app.main:app --reload
```
The API will be available at:

**API:** http://127.0.0.1:8000

### 5. Verify the Backend

Use the health endpoint to confirm that the backend is running correctly.

**Health Check:** http://127.0.0.1:8000/health

The model information endpoint provides runtime information about the loaded model and configuration.

**Model Information:** http://127.0.0.1:8000/model-info

### 6. Explore the API with Swagger

FastAPI provides interactive API documentation through Swagger UI.

Open:

**Swagger UI:** http://127.0.0.1:8000/docs

Swagger can be used to inspect and test the available API endpoints directly from the browser.

The API provides functionality for:

- Health checks
- Model information
- Single-transaction prediction
- Batch prediction
- Monitoring
- Drift analysis
- Performance monitoring

### 7. Start the Streamlit Frontend

The Streamlit application provides the interactive user interface for the fraud detection platform.

Keep the FastAPI backend running and open a **second terminal**.

Activate the virtual environment again if required.

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```
**macOS / Linux**

```bash
source .venv/bin/activate
```
Start the Streamlit application:

```bash
streamlit run streamlit/app.py
```
The frontend will normally be available at:

**Streamlit:** http://localhost:8501

### 8. Use the Platform

Once both services are running, the platform provides the following workflows:

- **Fraud Prediction** — evaluate an individual transaction using the finalized fraud detection model.
- **Batch Prediction** — process multiple transactions through the reusable inference API.
- **Monitoring** — review available prediction, drift, and performance information.

The Streamlit frontend communicates with the FastAPI backend, while the backend handles model loading, inference, and monitoring operations.

### 9. Run the Test Suite

The repository includes automated tests covering the application's core functionality and API behavior.

Run the test suite from the project root:

```bash
pytest -q
```
A successful test run confirms that the current application components and API contracts are functioning as expected.

### Local Runtime Architecture

The local platform consists of two coordinated application processes:

```text
┌─────────────────────────┐
│   Streamlit Frontend    │
│       Port 8501         │
└────────────┬────────────┘
             │
             │ HTTP Requests
             ▼
┌─────────────────────────┐
│     FastAPI Backend     │
│       Port 8000         │
├─────────────────────────┤
│ Prediction API          │
│ Batch Prediction API    │
│ Model Information       │
│ Monitoring & Drift      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Final Model Artifact  │
│     + Model Metadata    │
└─────────────────────────┘
```
This separation keeps the inference layer independent from the user interface and allows the FastAPI service to be consumed by Streamlit or other API clients.

### Complete Local Startup

For the complete platform experience, run the services in two separate terminals.

**Terminal 1 — FastAPI Backend**

```bash
uvicorn app.main:app --reload
```
**Terminal 2 — Streamlit Frontend**

```bash
streamlit run streamlit/app.py
```
Then access the application at:

**Streamlit:** http://localhost:8501

For direct API interaction and endpoint testing:

**Swagger UI:** http://127.0.0.1:8000/docs

### Development Notes

This setup is intended for running the finalized inference platform using the existing model artifact and metadata.

Model training, hyperparameter tuning, notebook execution, and experimentation are development workflows and are not required for normal application usage.

## Swaransh Mishra

**Data Analyst • Business Analyst • Data Scientist**

### Connect with Me

<p align="left">

<a href="https://github.com/Swaransh-Mishra">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github"/>
</a>

<a href="https://www.linkedin.com/in/swaransh-mishra-a85123258/">
<img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin"/>
</a>

<a href="mailto:swaransh03122003@gmail.com">
<img src="https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail"/>
</a>

</p>

## License

This project is licensed under the **MIT License**.
