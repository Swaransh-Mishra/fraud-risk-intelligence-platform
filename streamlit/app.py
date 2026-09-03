import numpy as np
import pandas as pd
import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Fraud Risk Intelligence Platform",
    page_icon="🔐",
    layout="wide",
)


st.html(
    """
    <style>
        .stApp {
            background: #0d1117;
        }

        [data-testid="stSidebar"] {
            background: #24242f;
            border-right: 1px solid #30303c;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 2rem;
        }

        [data-testid="stSidebar"] h1 {
            color: #f4f4f5;
            font-size: 1.25rem;
            margin-bottom: 1.5rem;
        }

        [data-testid="stSidebar"] label {
            color: #d4d4d8;
        }

        .main-title {
            font-size: 2.25rem;
            font-weight: 700;
            color: #f4f4f5;
            margin-bottom: 0.25rem;
            letter-spacing: -0.02em;
        }

        .page-title {
            font-size: 2rem;
            font-weight: 700;
            color: #f4f4f5;
            margin-bottom: 0.25rem;
            letter-spacing: -0.02em;
        }

        .page-subtitle {
            color: #8b949e;
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
        }

        .section-title {
            color: #f4f4f5;
            font-size: 1.05rem;
            font-weight: 600;
            margin: 0.4rem 0 1rem 0;
        }

        .metric-card {
            background: #151a21;
            border: 1px solid #292f38;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            min-height: 92px;
            box-sizing: border-box;
        }

        .metric-label {
            color: #8b949e;
            font-size: 0.75rem;
            margin-bottom: 0.45rem;
        }

        .metric-value {
            color: #f4f4f5;
            font-size: 1.55rem;
            font-weight: 600;
            line-height: 1.1;
        }

        .model-card {
            background: #151a21;
            border: 1px solid #292f38;
            border-radius: 8px;
            padding: 1.15rem 1.25rem;
            min-height: 190px;
            box-sizing: border-box;
        }

        .model-name {
            color: #f4f4f5;
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        .model-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.45rem 0;
            border-bottom: 1px solid #242a32;
            font-size: 0.82rem;
        }

        .model-row:last-child {
            border-bottom: none;
        }

        .model-key {
            color: #8b949e;
        }

        .model-value {
            color: #d4d4d8;
            text-align: right;
        }

        .result-card {
            background: #151a21;
            border: 1px solid #292f38;
            border-radius: 8px;
            padding: 1.1rem 1.2rem;
            margin-top: 0.5rem;
            box-sizing: border-box;
        }

        .result-fraud {
            border-left: 4px solid #ff4b4b;
        }

        .result-safe {
            border-left: 4px solid #3fb950;
        }

        .result-title {
            color: #f4f4f5;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.3rem;
        }

        .result-text {
            color: #8b949e;
            font-size: 0.82rem;
        }

        .upload-card {
            background: #151a21;
            border: 1px solid #292f38;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            box-sizing: border-box;
        }

        .upload-title {
            color: #f4f4f5;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }

        .small-note {
            color: #6e7681;
            font-size: 0.75rem;
        }

        .feature-group {
            color: #c9d1d9;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin: 0.5rem 0 0.8rem 0;
        }

        div[data-testid="stMetric"] {
            background: #151a21;
            border: 1px solid #292f38;
            padding: 0.8rem 1rem;
            border-radius: 8px;
        }

        div[data-testid="stMetricLabel"] {
            color: #8b949e;
        }

        div[data-testid="stMetricValue"] {
            color: #f4f4f5;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #292f38;
            border-radius: 8px;
            overflow: hidden;
        }

        .stButton > button {
            border-radius: 6px;
            font-weight: 600;
            min-height: 2.5rem;
        }

        .stButton > button[kind="primary"] {
            background: #ff4b4b;
            border: 1px solid #ff4b4b;
        }

        .stButton > button[kind="primary"]:hover {
            background: #ff5c5c;
            border-color: #ff5c5c;
        }

        div[data-baseweb="input"],
        div[data-baseweb="select"] {
            border-radius: 6px;
        }

        hr {
            border-color: #292f38 !important;
        }
    </style>
    """
)


def get_api_data(endpoint):
    response = requests.get(
        f"{API_BASE_URL}{endpoint}",
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


def post_api_data(endpoint, payload):
    response = requests.post(
        f"{API_BASE_URL}{endpoint}",
        json=payload,
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def show_api_error(exc, action):
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            detail = exc.response.json().get(
                "detail",
                "Unknown API error",
            )
        except ValueError:
            detail = exc.response.text

        st.error(f"{action} failed: {detail}")
    else:
        st.error(f"{action} failed: {exc}")


def clean_batch_data(df, required_features):
    df = df[required_features].copy()

    non_negative_features = [
        "TX_AMOUNT",
        "customer_tx_count",
        "customer_avg_amount",
        "customer_max_amount",
        "customer_amount_std",
        "time_since_customer_tx",
        "customer_tx_count_1h",
        "customer_tx_count_24h",
        "customer_amount_sum_24h",
        "terminal_tx_count",
        "terminal_avg_amount",
        "terminal_max_amount",
        "terminal_amount_std",
        "terminal_fraud_count",
        "terminal_tx_count_1h",
        "terminal_tx_count_24h",
        "terminal_amount_sum_24h",
    ]

    for feature in non_negative_features:
        if feature in df.columns:
            tiny_negative = (
                (df[feature] < 0)
                & np.isclose(
                    df[feature],
                    0,
                    atol=1e-10,
                )
            )

            df.loc[tiny_negative, feature] = 0.0

    return df


def show_header(title, subtitle, main=False):
    title_class = "main-title" if main else "page-title"

    st.html(
        f"""
        <div class="{title_class}">
            {title}
        </div>

        <div class="page-subtitle">
            {subtitle}
        </div>
        """
    )


def show_section_title(title):
    st.html(
        f"""
        <div class="section-title">
            {title}
        </div>
        """
    )


def show_metric_card(label, value):
    st.html(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {label}
            </div>

            <div class="metric-value">
                {value}
            </div>
        </div>
        """
    )


def show_model_card(model_info):
    st.html(
        f"""
        <div class="model-card">

            <div class="model-name">
                {model_info["model_name"]}
            </div>

            <div class="model-row">
                <span class="model-key">
                    Model Type
                </span>

                <span class="model-value">
                    {model_info["model_type"]}
                </span>
            </div>

            <div class="model-row">
                <span class="model-key">
                    Features
                </span>

                <span class="model-value">
                    {model_info["feature_count"]}
                </span>
            </div>

            <div class="model-row">
                <span class="model-key">
                    Training Records
                </span>

                <span class="model-value">
                    {model_info["training_records"]:,}
                </span>
            </div>

            <div class="model-row">
                <span class="model-key">
                    Decision Threshold
                </span>

                <span class="model-value">
                    {model_info["decision_threshold"]:.2f}
                </span>
            </div>

        </div>
        """
    )


def show_result_card(predicted_fraud):
    if predicted_fraud:
        st.html(
            """
            <div class="result-card result-fraud">

                <div class="result-title">
                    High fraud risk detected
                </div>

                <div class="result-text">
                    The production model classified this
                    transaction as potentially fraudulent.
                </div>

            </div>
            """
        )

    else:
        st.html(
            """
            <div class="result-card result-safe">

                <div class="result-title">
                    Transaction classified as non-fraudulent
                </div>

                <div class="result-text">
                    The production model did not identify
                    this transaction as fraudulent.
                </div>

            </div>
            """
        )


FEATURE_CONFIG = {
    "Transaction Details": [
        {
            "name": "TX_AMOUNT",
            "label": "Transaction Amount",
            "type": "float",
            "min_value": 0.0,
            "value": 150.0,
        },
        {
            "name": "hour_of_day",
            "label": "Hour of Day",
            "type": "int",
            "min_value": 0,
            "max_value": 23,
            "value": 14,
        },
        {
            "name": "day_of_week",
            "label": "Day of Week",
            "type": "int",
            "min_value": 0,
            "max_value": 6,
            "value": 2,
        },
        {
            "name": "is_weekend",
            "label": "Weekend",
            "type": "select",
            "options": [0, 1],
        },
        {
            "name": "customer_tx_count",
            "label": "Customer Transactions",
            "type": "int",
            "min_value": 0,
            "value": 10,
        },
        {
            "name": "customer_avg_amount",
            "label": "Customer Average Amount",
            "type": "float",
            "min_value": 0.0,
            "value": 120.0,
        },
        {
            "name": "customer_max_amount",
            "label": "Customer Maximum Amount",
            "type": "float",
            "min_value": 0.0,
            "value": 200.0,
        },
        {
            "name": "customer_amount_std",
            "label": "Customer Amount Std",
            "type": "float",
            "min_value": 0.0,
            "value": 30.0,
        },
        {
            "name": "time_since_customer_tx",
            "label": "Time Since Customer Transaction",
            "type": "float",
            "min_value": 0.0,
            "value": 3600.0,
        },
        {
            "name": "customer_amount_deviation",
            "label": "Customer Amount Deviation",
            "type": "float",
            "value": 30.0,
        },
        {
            "name": "customer_amount_ratio",
            "label": "Customer Amount Ratio",
            "type": "float",
            "min_value": 0.0,
            "value": 1.25,
        },
        {
            "name": "customer_tx_count_1h",
            "label": "Customer Transactions (1h)",
            "type": "int",
            "min_value": 0,
            "value": 1,
        },
    ],
    "Customer Activity": [
        {
            "name": "customer_tx_count_24h",
            "label": "Customer Transactions (24h)",
            "type": "int",
            "min_value": 0,
            "value": 5,
        },
        {
            "name": "customer_amount_sum_24h",
            "label": "Customer Amount Sum (24h)",
            "type": "float",
            "min_value": 0.0,
            "value": 600.0,
        },
    ],
    "Terminal Activity": [
        {
            "name": "terminal_tx_count",
            "label": "Terminal Transactions",
            "type": "int",
            "min_value": 0,
            "value": 20,
        },
        {
            "name": "terminal_avg_amount",
            "label": "Terminal Average Amount",
            "type": "float",
            "min_value": 0.0,
            "value": 130.0,
        },
        {
            "name": "terminal_max_amount",
            "label": "Terminal Maximum Amount",
            "type": "float",
            "min_value": 0.0,
            "value": 250.0,
        },
        {
            "name": "terminal_amount_std",
            "label": "Terminal Amount Std",
            "type": "float",
            "min_value": 0.0,
            "value": 40.0,
        },
        {
            "name": "terminal_fraud_count",
            "label": "Terminal Fraud Count",
            "type": "int",
            "min_value": 0,
            "value": 1,
        },
        {
            "name": "terminal_fraud_rate",
            "label": "Terminal Fraud Rate",
            "type": "float",
            "min_value": 0.0,
            "max_value": 1.0,
            "value": 0.05,
        },
        {
            "name": "terminal_tx_count_1h",
            "label": "Terminal Transactions (1h)",
            "type": "int",
            "min_value": 0,
            "value": 2,
        },
        {
            "name": "terminal_tx_count_24h",
            "label": "Terminal Transactions (24h)",
            "type": "int",
            "min_value": 0,
            "value": 10,
        },
        {
            "name": "terminal_amount_sum_24h",
            "label": "Terminal Amount Sum (24h)",
            "type": "float",
            "min_value": 0.0,
            "value": 1300.0,
        },
    ],
}


def render_feature_input(config):
    if config["type"] == "select":
        return st.selectbox(
            config["label"],
            options=config["options"],
            format_func=lambda x: "Yes" if x else "No",
        )

    if config["type"] == "int":
        kwargs = {
            "label": config["label"],
            "min_value": config.get("min_value", 0),
            "value": config["value"],
            "step": 1,
        }

        if "max_value" in config:
            kwargs["max_value"] = config["max_value"]

        return st.number_input(**kwargs)

    kwargs = {
        "label": config["label"],
        "value": config["value"],
    }

    if "min_value" in config:
        kwargs["min_value"] = config["min_value"]

    if "max_value" in config:
        kwargs["max_value"] = config["max_value"]

    return st.number_input(**kwargs)


def show_overview():
    show_header(
        "Fraud Risk Intelligence Platform",
        "ML-powered transaction fraud risk assessment",
        main=True,
    )

    try:
        health = get_api_data("/health")
        model_info = get_api_data("/model-info")
        monitoring = get_api_data("/monitoring")

        summary = monitoring["summary"]

        st.divider()

        show_section_title("Platform Overview")

        columns = st.columns(6)

        metrics = [
            (
                "Status",
                health["status"].title(),
            ),
            (
                "Transactions",
                f'{summary["total_transactions"]:,}',
            ),
            (
                "Fraud Alerts",
                f'{summary["total_predicted_fraud"]:,}',
            ),
            (
                "Alert Rate",
                f'{summary["fraud_alert_rate"]:.1%}',
            ),
            (
                "Avg. Fraud Probability",
                f'{summary["average_fraud_probability"]:.1%}',
            ),
            (
                "Decision Threshold",
                f'{model_info["decision_threshold"]:.2f}',
            ),
        ]

        for column, (label, value) in zip(
            columns,
            metrics,
        ):
            with column:
                show_metric_card(
                    label,
                    value,
                )

        st.divider()

        col1, col2 = st.columns(
            2,
            gap="large",
        )

        with col1:
            show_section_title("Production Model")
            show_model_card(model_info)

        with col2:
            show_section_title("Prediction Activity")

            activity = pd.DataFrame(
                monitoring["event_type_summary"]
            )

            if not activity.empty:
                display_columns = [
                    "event_type",
                    "event_count",
                    "total_transactions",
                    "predicted_fraud_count",
                ]

                st.dataframe(
                    activity[display_columns],
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.info(
                    "No prediction activity recorded yet."
                )

    except requests.RequestException as exc:
        show_api_error(
            exc,
            "Loading platform overview",
        )


def show_fraud_prediction():
    show_header(
        "Fraud Prediction",
        "Analyze the fraud risk of a single transaction.",
    )

    payload = {}

    for section_name, features in FEATURE_CONFIG.items():

        show_section_title(section_name)

        if section_name == "Transaction Details":
            columns = st.columns(3)
        elif section_name == "Customer Activity":
            columns = st.columns(2)
        else:
            columns = st.columns(3)

        for index, feature in enumerate(features):
            column = columns[index % len(columns)]

            with column:
                payload[feature["name"]] = render_feature_input(
                    feature
                )

    st.divider()

    if st.button(
        "Analyze Transaction",
        type="primary",
        use_container_width=True,
    ):
        try:
            result = post_api_data(
                "/predict",
                payload,
            )

            st.divider()

            show_section_title(
                "Fraud Risk Assessment"
            )

            columns = st.columns(4)

            with columns[0]:
                show_metric_card(
                    "Fraud Probability",
                    f'{result["fraud_probability"]:.2%}',
                )

            with columns[1]:
                show_metric_card(
                    "Risk Score",
                    f'{result["fraud_risk_score"]:.2f}',
                )

            with columns[2]:
                prediction = (
                    "FRAUD"
                    if result["predicted_fraud"]
                    else "NOT FRAUD"
                )

                show_metric_card(
                    "Prediction",
                    prediction,
                )

            with columns[3]:
                show_metric_card(
                    "Risk Level",
                    result["risk_level"].upper(),
                )

            show_result_card(
                result["predicted_fraud"]
            )

        except requests.RequestException as exc:
            show_api_error(
                exc,
                "Prediction request",
            )


def show_batch_prediction():
    show_header(
        "Batch Prediction",
        "Analyze multiple transactions using the production fraud model.",
    )

    try:
        model_info = get_api_data("/model-info")
        required_features = model_info["features"]

    except requests.RequestException as exc:
        show_api_error(
            exc,
            "Loading model information",
        )
        return

    st.html(
        f"""
        <div class="upload-card">

            <div class="upload-title">
                Upload transaction CSV
            </div>

            <div class="small-note">
                Required model features: {len(required_features)}
                &nbsp;·&nbsp;
                Maximum batch size: 1,000 transactions
            </div>

        </div>
        """
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info(
            f"Upload a CSV file containing "
            f"{len(required_features)} model features."
        )
        return

    try:
        transactions = pd.read_csv(
            uploaded_file
        )

        missing_features = [
            feature
            for feature in required_features
            if feature not in transactions.columns
        ]

        if missing_features:
            st.error(
                "Missing required features: "
                + ", ".join(missing_features)
            )
            return

        transactions = clean_batch_data(
            transactions,
            required_features,
        )

        if transactions.empty:
            st.warning(
                "The uploaded CSV does not contain any transactions."
            )
            return

        if len(transactions) > 1000:
            st.error(
                "Batch prediction supports a maximum "
                "of 1000 transactions."
            )
            return

        show_section_title(
            "Transaction Preview"
        )

        st.dataframe(
            transactions.head(10),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"{len(transactions):,} transactions ready for prediction."
        )

        st.divider()

        if st.button(
            "Run Batch Prediction",
            type="primary",
            use_container_width=True,
        ):
            payload = {
                "transactions": transactions.to_dict(
                    orient="records"
                )
            }

            try:
                result = post_api_data(
                    "/predict/batch",
                    payload,
                )

                show_section_title(
                    "Batch Prediction Results"
                )

                columns = st.columns(4)

                with columns[0]:
                    show_metric_card(
                        "Transactions",
                        f'{result["total_transactions"]:,}',
                    )

                with columns[1]:
                    show_metric_card(
                        "Fraud Alerts",
                        f'{result["predicted_fraud_count"]:,}',
                    )

                with columns[2]:
                    alert_rate = (
                        result["predicted_fraud_count"]
                        / result["total_transactions"]
                    )

                    show_metric_card(
                        "Alert Rate",
                        f"{alert_rate:.1%}",
                    )

                with columns[3]:
                    show_metric_card(
                        "Avg. Fraud Probability",
                        f'{result["average_fraud_probability"]:.2%}',
                    )

                results = pd.DataFrame(
                    result["predictions"]
                )

                results.insert(
                    0,
                    "transaction_id",
                    range(
                        1,
                        len(results) + 1,
                    ),
                )

                results["predicted_fraud"] = (
                    results["predicted_fraud"].map(
                        {
                            0: "Not Fraud",
                            1: "Fraud",
                        }
                    )
                )

                results["fraud_probability"] = (
                    results["fraud_probability"] * 100
                ).round(2)

                results["fraud_risk_score"] = (
                    results["fraud_risk_score"].round(2)
                )

                st.dataframe(
                    results,
                    use_container_width=True,
                    hide_index=True,
                )

                csv_data = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Prediction Results",
                    data=csv_data,
                    file_name="fraud_prediction_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except requests.RequestException as exc:
                show_api_error(
                    exc,
                    "Batch prediction request",
                )

    except Exception as exc:
        st.error(
            f"Unable to read the uploaded CSV: {exc}"
        )


def show_monitoring():
    show_header(
        "Monitoring",
        "Monitor prediction activity and production model usage.",
    )

    try:
        monitoring = get_api_data(
            "/monitoring"
        )

        summary = monitoring["summary"]

        st.divider()

        show_section_title(
            "Monitoring Overview"
        )

        columns = st.columns(5)

        metrics = [
            (
                "Prediction Events",
                f'{summary["total_events"]:,}',
            ),
            (
                "Transactions",
                f'{summary["total_transactions"]:,}',
            ),
            (
                "Fraud Alerts",
                f'{summary["total_predicted_fraud"]:,}',
            ),
            (
                "Alert Rate",
                f'{summary["fraud_alert_rate"]:.1%}',
            ),
            (
                "Avg. Fraud Probability",
                f'{summary["average_fraud_probability"]:.2%}',
            ),
        ]

        for column, (label, value) in zip(
            columns,
            metrics,
        ):
            with column:
                show_metric_card(
                    label,
                    value,
                )

        st.divider()

        col1, col2 = st.columns(
            2,
            gap="large",
        )

        with col1:
            show_section_title(
                "Event Type Summary"
            )

            event_summary = pd.DataFrame(
                monitoring["event_type_summary"]
            )

            if not event_summary.empty:
                st.dataframe(
                    event_summary,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "No monitoring events recorded yet."
                )

        with col2:
            show_section_title(
                "Model Activity"
            )

            model_summary = pd.DataFrame(
                monitoring["model_activity_summary"]
            )

            if not model_summary.empty:
                st.dataframe(
                    model_summary,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "No model activity recorded yet."
                )

        st.divider()

        show_section_title(
            "Latest Prediction Events"
        )

        latest_events = pd.DataFrame(
            monitoring["latest_events"]
        )

        if not latest_events.empty:

            display_columns = [
                "timestamp",
                "event_type",
                "model_name",
                "decision_threshold",
                "total_transactions",
                "predicted_fraud_count",
                "average_fraud_probability",
            ]

            st.dataframe(
                latest_events[display_columns],
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info(
                "No recent prediction events."
            )

    except requests.RequestException as exc:
        show_api_error(
            exc,
            "Loading monitoring data",
        )


st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Fraud Prediction",
        "Batch Prediction",
        "Monitoring",
    ],
)


if page == "Overview":
    show_overview()

elif page == "Fraud Prediction":
    show_fraud_prediction()

elif page == "Batch Prediction":
    show_batch_prediction()

elif page == "Monitoring":
    show_monitoring()