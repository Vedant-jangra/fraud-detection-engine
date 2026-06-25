import streamlit as st
import requests
import os
import time

# Configure page
st.set_page_config(page_title="Fraud Intelligence Engine", page_icon="🛡️", layout="wide")

# Custom CSS for modern styling
st.markdown(
    """
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #1e212b;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .high-risk {
        color: #ff4b4b;
        font-weight: bold;
    }
    .low-risk {
        color: #00cc96;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🛡️ Real-Time Fraud Intelligence Engine")
st.markdown("Enter transaction details below to evaluate the fraud risk in real-time.")

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Transaction Details")
    with st.form("transaction_form"):
        transaction_id = st.text_input("Transaction ID", value="txn-demo-001")
        user_id = st.text_input("User ID", value="user_00001")
        timestamp = st.text_input("Timestamp (ISO 8601)", value="2026-01-15T14:30:00Z")
        amount = st.number_input("Amount ($)", value=45000.00, min_value=0.0)
        merchant_cat = st.selectbox(
            "Merchant Category",
            [
                "electronics",
                "grocery",
                "travel",
                "restaurant",
                "utility",
                "crypto",
                "unknown",
            ],
        )
        country = st.selectbox("Country Code", ["AE", "US", "IN", "SG", "XX"])

        submitted = st.form_submit_button("Score Transaction", use_container_width=True)

with col2:
    if submitted:
        payload = {
            "transaction_id": transaction_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "amount": amount,
            "merchant_cat": merchant_cat,
            "country": country,
        }

        with st.spinner("Scoring transaction..."):
            try:
                # We call the /v1/explain endpoint to get both score and SHAP explanations
                start_time = time.time()
                response = requests.post(
                    f"{API_URL}/v1/explain", json=payload, timeout=5
                )
                end_time = time.time()

                if response.status_code == 200:
                    data = response.json()

                    st.subheader("Risk Assessment")

                    # Metrics Row
                    m1, m2, m3 = st.columns(3)

                    prob = data["fraud_probability"]
                    risk_level = data["risk_level"]
                    latency = data.get("latency_ms", (end_time - start_time) * 1000)

                    m1.metric("Fraud Probability", f"{prob:.2%}")

                    risk_color = (
                        "🔴"
                        if risk_level == "HIGH"
                        else ("🟡" if risk_level == "MEDIUM" else "🟢")
                    )
                    m2.metric("Risk Level", f"{risk_color} {risk_level}")

                    m3.metric("Latency", f"{latency:.2f} ms")

                    st.markdown("---")

                    # SHAP Explanations
                    if "explanation" in data and "top_factors" in data["explanation"]:
                        st.subheader("Why was this decision made?")
                        st.markdown(
                            f"**Explanation Method:** {data['explanation'].get('explainability_method', 'SHAP TreeExplainer')}"
                        )

                        factors = data["explanation"]["top_factors"]

                        for factor in factors:
                            # Display direction with arrows
                            direction = factor.get("direction", "")
                            feature = factor.get("feature", "unknown")
                            shap_val = factor.get("shap_value", 0.0)

                            if "increases" in direction.lower():
                                st.error(
                                    f"⬆️ **{feature}** strongly increased fraud risk (SHAP: {shap_val:.4f})"
                                )
                            else:
                                st.success(
                                    f"⬇️ **{feature}** decreased fraud risk (SHAP: {shap_val:.4f})"
                                )

                    else:
                        st.info(
                            "No detailed explanation available for this transaction."
                        )

                else:
                    st.error(f"API Error: {response.status_code}")
                    st.json(response.text)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the FastAPI Backend. Is the API running?"
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info(
            "👈 Fill out the transaction details and click **Score Transaction** to begin."
        )

st.markdown("---")
st.markdown("*Architecture: Kafka → PySpark → TimescaleDB → FastAPI → ONNX Runtime*")
