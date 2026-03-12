"""
Streamlit Frontend for Loan Approval Prediction System
Communicates with the FastAPI backend via HTTP requests.
"""
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Helper functions to call the FastAPI backend
# -------------------------------------------------------------------

def api_health_check() -> dict | None:
    """Check if the backend API is healthy."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Health check failed: {e}")
        return None


def api_predict_single(data: dict) -> dict | None:
    """Call the /predict endpoint for a single prediction."""
    try:
        resp = requests.post(f"{API_BASE_URL}/predict", json=data, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend API. Make sure the FastAPI server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The server may be overloaded.")
        return None
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", e.response.text)
        except Exception:
            detail = e.response.text
        st.error(f"API Error ({e.response.status_code}): {detail}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        return None


def api_batch_predict(file_bytes: bytes, filename: str) -> dict | None:
    """Call the /batch_predict endpoint with a CSV file."""
    try:
        files = {"file": (filename, file_bytes, "text/csv")}
        resp = requests.post(f"{API_BASE_URL}/batch_predict", files=files, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend API. Make sure the FastAPI server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("Batch prediction timed out. Try a smaller CSV file.")
        return None
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", e.response.text)
        except Exception:
            detail = e.response.text
        st.error(f"API Error ({e.response.status_code}): {detail}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        return None


def api_model_info() -> dict | None:
    """Call the /model_info endpoint."""
    try:
        resp = requests.get(f"{API_BASE_URL}/model_info", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Model info request failed: {e}")
        return None


def api_explain(data: dict) -> dict | None:
    """Call the /explain endpoint for SHAP values."""
    try:
        resp = requests.post(f"{API_BASE_URL}/explain", json=data, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Explain request failed: {e}")
        return None


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
# Title
st.markdown('<h1 class="main-header">🏦 Loan Approval Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.title("Navigation")

    page = st.radio(
        "Select Page",
        ["🏠 Home", "🔮 Single Prediction", "📊 Batch Prediction", "📈 Model Info"]
    )

    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This application predicts loan approval probability "
        "using machine learning algorithms trained on historical data."
    )

    st.markdown("---")
    st.markdown("### Tech Stack")
    st.markdown("- **ML**: XGBoost")
    st.markdown("- **Explainability**: SHAP")
    st.markdown("- **API**: FastAPI")
    st.markdown("- **Frontend**: Streamlit")

    # API connection status
    st.markdown("---")
    health = api_health_check()
    if health:
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Offline — start the backend first")

# -------------------------------------------------------------------
# Home Page
# -------------------------------------------------------------------
if page == "🏠 Home":
    st.header("Welcome to Loan Approval Prediction System")

    # Fetch real metrics from the API
    info = api_model_info()
    metrics = info.get("metrics") if info else None

    col1, col2, col3 = st.columns(3)

    with col1:
        model_name = info.get("model_type", "N/A") if info else "N/A"
        st.metric(
            label="Model Algorithm",
            value=model_name
        )

    with col2:
        if metrics:
            auc = metrics.get("roc_auc", "N/A")
            st.metric(label="ROC-AUC Score", value=f"{auc}")
        else:
            st.metric(label="ROC-AUC Score", value="N/A")

    with col3:
        if metrics:
            test_samples = metrics.get("test_samples", "N/A")
            train_samples = metrics.get("train_samples", "N/A")
            st.metric(label="Training Samples", value=f"{train_samples:,}" if isinstance(train_samples, int) else str(train_samples))
        else:
            st.metric(label="Training Samples", value="N/A")

    st.markdown("---")

    # Features
    st.subheader("📋 Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Capabilities
        - Single customer prediction
        - Batch prediction via CSV upload
        - Real-time probability calculation
        - Risk level categorization
        - SHAP feature importance visualization
        """)

    with col2:
        st.markdown("""
        ### Model Features
        - Credit utilization ratio
        - Payment history (30, 60, 90 days late)
        - Debt-to-income ratio
        - Number of credit lines
        - Monthly income
        - Age and dependents
        """)

    st.markdown("---")

    # How It Works
    st.subheader("🔍 How It Works")

    st.markdown("""
    1. **Input Data**: Provide borrower information
    2. **Feature Engineering**: System creates additional features
    3. **Model Prediction**: XGBoost model predicts default probability
    4. **Risk Assessment**: Categorizes risk level (Low/Medium/High)
    5. **Results**: Get detailed prediction with probabilities
    """)

# -------------------------------------------------------------------
# Single Prediction Page
# -------------------------------------------------------------------
elif page == "🔮 Single Prediction":
    st.header("Single Customer Prediction")
    st.markdown("Enter customer details to predict loan approval probability")

    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=120, value=45, step=1)
            monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=500000, value=9120, step=100)
            debt_ratio = st.number_input("Debt Ratio", min_value=0.0, max_value=50000.0, value=0.80, step=0.01, format="%.2f")
            credit_util = st.number_input("Credit Utilization", min_value=0.0, max_value=50.0, value=0.77, step=0.01, format="%.2f")
            num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=2, step=1)

        with col2:
            num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, max_value=100, value=13, step=1)
            num_real_estate = st.number_input("Real Estate Loans", min_value=0, max_value=50, value=6, step=1)
            times_30_59_late = st.number_input("Times 30-59 Days Late", min_value=0, max_value=98, value=2, step=1)
            times_60_89_late = st.number_input("Times 60-89 Days Late", min_value=0, max_value=98, value=0, step=1)
            times_90_late = st.number_input("Times 90+ Days Late", min_value=0, max_value=98, value=0, step=1)

        submitted = st.form_submit_button("🔮 Predict Default Probability", use_container_width=True)

    if submitted:
        # Prepare input data (use alias names for the API)
        input_data = {
            "RevolvingUtilizationOfUnsecuredLines": credit_util,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": times_30_59_late,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": num_credit_lines,
            "NumberOfTimes90DaysLate": times_90_late,
            "NumberRealEstateLoansOrLines": num_real_estate,
            "NumberOfTime60-89DaysPastDueNotWorse": times_60_89_late,
            "NumberOfDependents": num_dependents
        }

        # Make prediction via API
        with st.spinner("Making prediction..."):
            result = api_predict_single(input_data)

        if result:
            st.success("Prediction Complete!")

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Prediction",
                    value=result['prediction_label']
                )

            with col2:
                st.metric(
                    label="Default Probability",
                    value=f"{result['probability_default']:.2%}"
                )

            with col3:
                st.metric(
                    label="Risk Level",
                    value=result['risk_level']
                )

            # Probability gauge
            try:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=result['probability_default'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Default Probability (%)"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 60
                        }
                    }
                ))

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render gauge chart: {e}")

            # Risk interpretation
            if result['risk_level'] == 'Low Risk':
                st.markdown('<div class="success-box"><strong>Low Risk:</strong> This borrower has a low probability of default. Recommend approval.</div>', unsafe_allow_html=True)
            elif result['risk_level'] == 'Medium Risk':
                st.markdown('<div class="warning-box"><strong>Medium Risk:</strong> This borrower has moderate default risk. Further review recommended.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="danger-box"><strong>High Risk:</strong> This borrower has a high probability of default. Recommend rejection or additional collateral.</div>', unsafe_allow_html=True)

            # SHAP Explanation
            st.markdown("---")
            st.subheader("🔍 Why this prediction? (SHAP Feature Importance)")
            with st.spinner("Calculating feature contributions..."):
                explanation = api_explain(input_data)

            if explanation:
                features = explanation['features']
                shap_vals = explanation['shap_values']

                # Color: positive SHAP = pushes toward default (red), negative = away (green)
                colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_vals]

                fig = go.Figure(go.Bar(
                    x=shap_vals,
                    y=features,
                    orientation='h',
                    marker_color=colors
                ))
                fig.update_layout(
                    title="Feature Contributions to Default Prediction",
                    xaxis_title="SHAP Value (impact on prediction)",
                    yaxis_title="Feature",
                    height=400,
                    yaxis=dict(autorange="reversed"),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Red bars push toward **Default**, green bars push toward **No Default**.")
            else:
                st.info("SHAP explanations unavailable. The API may not have the shap package installed.")

# -------------------------------------------------------------------
# Batch Prediction Page
# -------------------------------------------------------------------
elif page == "📊 Batch Prediction":
    st.header("Batch Prediction")
    st.markdown("Upload a CSV file with customer data for batch predictions")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with customer data. Must contain all required columns."
    )

    if uploaded_file is not None:
        # Read CSV for preview
        try:
            df = pd.read_csv(uploaded_file)
        except pd.errors.EmptyDataError:
            st.error("Uploaded CSV file is empty.")
            st.stop()
        except pd.errors.ParserError as e:
            st.error(f"Failed to parse CSV file: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")
            st.stop()

        if df.empty:
            st.warning("The uploaded CSV has no data rows.")
            st.stop()

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown(f"**Total rows:** {len(df)}")

        # Validate columns
        required_columns = [
            'RevolvingUtilizationOfUnsecuredLines',
            'age',
            'NumberOfTime30-59DaysPastDueNotWorse',
            'DebtRatio',
            'MonthlyIncome',
            'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate',
            'NumberRealEstateLoansOrLines',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfDependents'
        ]

        missing_cols = set(required_columns) - set(df.columns)

        if missing_cols:
            st.error(f"Missing required columns: {sorted(missing_cols)}")
        else:
            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                with st.spinner("Making predictions..."):
                    # Reset file pointer and send raw bytes to API
                    uploaded_file.seek(0)
                    file_bytes = uploaded_file.read()
                    api_result = api_batch_predict(file_bytes, uploaded_file.name)

                if api_result:
                    predictions_list = api_result.get('predictions', [])
                    if not predictions_list:
                        st.warning("API returned no predictions.")
                        st.stop()

                    st.success("Batch prediction complete!")

                    # Build results dataframe
                    pred_df = pd.DataFrame(predictions_list)
                    result_df = df.copy()
                    result_df['Prediction'] = pred_df['prediction']
                    result_df['Prediction_Label'] = pred_df['prediction_label']
                    result_df['Default_Probability'] = pred_df['probability_default']
                    result_df['Risk_Level'] = pred_df['risk_level']

                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Predictions", len(result_df))

                    with col2:
                        default_count = (result_df['Prediction'] == 1).sum()
                        st.metric("Predicted Defaults", default_count)

                    with col3:
                        default_rate = (result_df['Prediction'] == 1).mean() * 100
                        st.metric("Default Rate", f"{default_rate:.1f}%")

                    with col4:
                        avg_prob = result_df['Default_Probability'].mean() * 100
                        st.metric("Avg Default Prob", f"{avg_prob:.1f}%")

                    # Visualizations
                    col1, col2 = st.columns(2)

                    with col1:
                        try:
                            fig = px.pie(
                                values=result_df['Prediction_Label'].value_counts().values,
                                names=result_df['Prediction_Label'].value_counts().index,
                                title="Prediction Distribution",
                                color_discrete_sequence=['#2ecc71', '#e74c3c']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not render pie chart: {e}")

                    with col2:
                        try:
                            risk_order = ['Low Risk', 'Medium Risk', 'High Risk']
                            risk_counts = result_df['Risk_Level'].value_counts()
                            risk_counts = risk_counts.reindex(risk_order).dropna().astype(int)
                            fig = px.bar(
                                x=risk_counts.index,
                                y=risk_counts.values,
                                title="Risk Level Distribution",
                                labels={'x': 'Risk Level', 'y': 'Count'},
                                color=risk_counts.index,
                                color_discrete_map={
                                    'Low Risk': 'green',
                                    'Medium Risk': 'orange',
                                    'High Risk': 'red'
                                }
                            )
                            fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not render bar chart: {e}")

                    # Results table
                    st.subheader("📊 Prediction Results")
                    st.dataframe(result_df, use_container_width=True)

                    # Download button
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# -------------------------------------------------------------------
# Model Info Page
# -------------------------------------------------------------------
elif page == "📈 Model Info":
    st.header("Model Information")

    info = api_model_info()

    if info:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model Details")
            st.markdown(f"""
            - **Model Type:** {info.get('model_type', 'N/A')}
            - **Model Path:** `{info.get('model_path', 'N/A')}`
            - **Features:** {info.get('features_count', 'N/A')}
            """)

        with col2:
            st.subheader("Performance Metrics")
            metrics = info.get('metrics')
            if metrics:
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
                    st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                with m_col2:
                    st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                    st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")

                # Confusion matrix if available
                if all(k in metrics for k in ('tp', 'tn', 'fp', 'fn')):
                    st.markdown("---")
                    st.subheader("Confusion Matrix")
                    cm_df = pd.DataFrame(
                        [[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]],
                        index=['Actual: No Default', 'Actual: Default'],
                        columns=['Predicted: No Default', 'Predicted: Default']
                    )
                    st.dataframe(cm_df, use_container_width=True)
            else:
                st.warning("metrics.json not found. Run the notebook to generate training metrics.")
    else:
        st.warning("Could not connect to the backend API. Start the FastAPI server first.")

    # Sample data
    st.subheader("📝 Sample Input Format")
    sample_df = pd.DataFrame({
        'RevolvingUtilizationOfUnsecuredLines': [0.77],
        'age': [45],
        'NumberOfTime30-59DaysPastDueNotWorse': [2],
        'DebtRatio': [0.80],
        'MonthlyIncome': [9120],
        'NumberOfOpenCreditLinesAndLoans': [13],
        'NumberOfTimes90DaysLate': [0],
        'NumberRealEstateLoansOrLines': [6],
        'NumberOfTime60-89DaysPastDueNotWorse': [0],
        'NumberOfDependents': [2]
    })
    st.dataframe(sample_df, use_container_width=True)

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>🏦 Loan Approval Prediction System | Built with FastAPI & Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
