import io
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
# =====================================================
# CONFIG
# =====================================================
API_URL = "http://127.0.0.1:8000"
TIMEOUT = 90

st.set_page_config(
    page_title="FraudShield AI Platform",
    layout="wide"
)

# =====================================================
# STATE
# =====================================================
if "scored_df" not in st.session_state:
    st.session_state.scored_df = None

# =====================================================
# HEADER
# =====================================================
st.title("💳 FraudShield AI Platform")

page = st.sidebar.radio(
    "Navigation",
    ["Prediction", "Batch Scoring", "Fraud Analytics", "Monitoring"]
)

# =====================================================
# 1. PREDICTION
# =====================================================
if page == "Prediction":

    st.subheader("🔎 Single Transaction Fraud Detection")

    col1, col2 = st.columns([2, 1])

    with col1:
        transaction_amount = st.number_input("Amount", min_value=0.0, value=100.0)
        hour = st.slider("Hour", 0, 23, 12)
        distance = st.number_input("Distance from home", min_value=0.0, value=5.0)
        tx_24h = st.number_input("Transactions (24h)", min_value=0, value=2)
        merchant_risk = st.slider("Merchant risk", 0.0, 1.0, 0.2)

        is_international = st.selectbox("International", [0, 1], index=0)
        is_card_present = st.selectbox("Card present", [0, 1], index=1)
        device_trust = st.slider("Device trust", 0.0, 1.0, 0.9)

        merchant_category = st.selectbox(
            "Merchant Category",
            ["grocery", "electronics", "travel", "restaurant", "gas", "retail", "healthcare", "entertainment", "utilities"]
        )
        merchant_country = st.selectbox(
            "Merchant Country",
            ["US", "CA", "GB", "DE", "NG", "IN", "FR", "BR"]
        )
        device_type = st.selectbox("Device Type", ["mobile", "web", "pos", "atm"])
        transaction_type = st.selectbox("Transaction Type", ["purchase", "transfer", "withdrawal", "deposit"])

        predict = st.button("🚀 Predict")

    with col2:
        st.markdown("### 💳 Card")
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #111827, #1f2937);
                color: white;
                border-radius: 18px;
                padding: 22px;
                min-height: 180px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.18);
            ">
                <div style="font-size: 16px; opacity: 0.9;">FraudShield Card</div>
                <div style="margin-top: 28px; font-size: 22px; letter-spacing: 2px;">•••• •••• •••• 1234</div>
                <div style="margin-top: 26px; font-size: 13px; opacity: 0.9;">CARD HOLDER</div>
                <div style="font-size: 18px;">Demo User</div>
                <div style="margin-top: 16px; font-size: 13px;">VALID THRU 12/28</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    customer_id = st.text_input("Customer ID", "C001")
    account_id = st.text_input("Account ID", "A001")

    if predict:
        payload = {
            "customer_id": customer_id,
            "account_id": account_id,
            "transaction_amount": transaction_amount,
            "merchant_category": merchant_category,
            "merchant_country": merchant_country,
            "device_type": device_type,
            "transaction_type": transaction_type,
            "hour": hour,
            "distance_from_home": distance,
            "transactions_last_24h": tx_24h,
            "merchant_risk_score": merchant_risk,
            "is_international": is_international,
            "is_card_present": is_card_present,
            "device_trust_score": device_trust,
            "account_balance": 2000.0,
        }

        try:
            res = requests.post(f"{API_URL}/predict_fraud", json=payload, timeout=TIMEOUT)

            if res.status_code == 200:
                r = res.json()

                c1, c2, c3 = st.columns(3)
                c1.metric("Fraud Probability", f"{r['fraud_probability']:.2%}")
                c2.metric("Prediction", r["predicted_label"].upper())
                c3.metric("Action", r["recommended_action"].upper())

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=r["fraud_probability"] * 100,
                    title={"text": "Fraud Risk"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "steps": [
                            {"range": [0, 40], "color": "green"},
                            {"range": [40, 70], "color": "yellow"},
                            {"range": [70, 100], "color": "red"},
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

                shap_data = r.get("shap_top_features", [])
                st.markdown("### 📊 Top SHAP Features")
                if shap_data:
                    shap_df = pd.DataFrame(shap_data).sort_values("shap_value")
                    fig2 = px.bar(
                        shap_df,
                        x="shap_value",
                        y="feature",
                        orientation="h",
                        title="Top Feature Impacts"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    st.dataframe(shap_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No SHAP data available")

            else:
                st.error(res.text)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# =====================================================
# 2. BATCH SCORING
# =====================================================
elif page == "Batch Scoring":

    st.subheader("📁 Upload CSV")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        preview_df = pd.read_csv(file)
        st.dataframe(preview_df.head(), use_container_width=True)

        if st.button("Run Batch"):
            try:
                file.seek(0)

                res = requests.post(
                    f"{API_URL}/predict_fraud_csv",
                    files={"file": file},
                    timeout=180
                )

                if res.status_code == 200:
                    r = res.json()
                    st.success(f"Scored {r['rows_scored']} rows")

                    scored = pd.DataFrame(r["preview"])
                    st.session_state.scored_df = scored
                    st.dataframe(scored, use_container_width=True)

                    download = requests.get(f"{API_URL}/download_report", timeout=TIMEOUT)
                    if download.status_code == 200:
                        st.download_button(
                            "⬇️ Download Full CSV",
                            data=download.content,
                            file_name="scored_transactions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Download file not ready yet")
                else:
                    st.error(res.text)

            except Exception as e:
                st.error(f"Batch scoring failed: {e}")

# =====================================================
# 3. FRAUD ANALYTICS
# =====================================================
elif page == "Fraud Analytics":

    st.subheader("📊 Fraud Analytics Dashboard")

    scored_upload = st.file_uploader(
        "Upload scored CSV (optional)",
        type=["csv"],
        key="scored_csv_uploader"
    )

    dashboard_df = None

    if scored_upload is not None:
        try:
            dashboard_df = pd.read_csv(scored_upload)
        except Exception as e:
            st.error(f"Could not read scored CSV: {e}")
    elif st.session_state.scored_df is not None:
        # load full latest scored file if available
        try:
            full_data = requests.get(f"{API_URL}/download_report", timeout=TIMEOUT)
            if full_data.status_code == 200:
                dashboard_df = pd.read_csv(io.BytesIO(full_data.content))
            else:
                dashboard_df = st.session_state.scored_df.copy()
        except Exception:
            dashboard_df = st.session_state.scored_df.copy()

    if dashboard_df is None:
        st.warning("Run batch scoring first, or upload a scored CSV.")
    else:
        required_cols = {"fraud_probability", "predicted_label", "risk_level"}
        missing_cols = required_cols - set(dashboard_df.columns)
        if missing_cols:
            st.error(f"Scored file is missing required columns: {sorted(missing_cols)}")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Transactions", len(dashboard_df))
            k2.metric("Avg Fraud Probability", f"{dashboard_df['fraud_probability'].mean():.2%}")
            k3.metric("Max Fraud Probability", f"{dashboard_df['fraud_probability'].max():.2%}")
            k4.metric(
                "Fraud Predictions",
                int((dashboard_df["predicted_label"].astype(str).str.lower() == "fraud").sum())
            )

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("### Risk Levels")
                risk = dashboard_df["risk_level"].astype(str).str.lower().value_counts().reset_index()
                risk.columns = ["risk", "count"]
                fig = px.pie(risk, names="risk", values="count")
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                st.markdown("### Predictions")
                pred = dashboard_df["predicted_label"].astype(str).str.lower().value_counts().reset_index()
                pred.columns = ["label", "count"]
                fig = px.pie(pred, names="label", values="count")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Fraud Distribution")
            fig = px.histogram(dashboard_df, x="fraud_probability", nbins=30)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Highest Risk Transactions")
            top_risk_df = dashboard_df.sort_values("fraud_probability", ascending=False).head(15)
            st.dataframe(top_risk_df, use_container_width=True, hide_index=True)

# =====================================================
# 4. MONITORING
# =====================================================
elif page == "Monitoring":

    st.subheader("📡 Monitoring Dashboard")

    try:
        metrics_res = requests.get(f"{API_URL}/metrics", timeout=TIMEOUT)
        data_res = requests.get(f"{API_URL}/monitoring_data", timeout=TIMEOUT)

        if metrics_res.status_code != 200:
            st.error("Failed to fetch metrics")
            st.stop()

        try:
            m = metrics_res.json()
        except Exception:
            st.error("Metrics endpoint returned invalid JSON")
            st.stop()

        if data_res.status_code != 200:
            st.error("Failed to fetch monitoring data")
            st.stop()

        try:
            d = data_res.json()
        except Exception:
            st.error("Monitoring endpoint returned invalid JSON")
            st.stop()

        records = d.get("records", [])

        # ==============================
        # 📊 KPI METRICS
        # ==============================
        c1, c2, c3 = st.columns(3)

        c1.metric("Total Predictions", m.get("total_predictions", 0))
        c2.metric("Avg Probability", f"{m.get('avg_probability', 0):.2f}")
        c3.metric("Fraud Rate", f"{m.get('fraud_rate', 0):.2%}")

        if records:
            df = pd.DataFrame(records)

            # 🔥 CLEAN DATA
            df = df.replace([np.nan, np.inf, -np.inf], 0)
            prob_col = "fraud_probability" if "fraud_probability" in df.columns else "probability"
            df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce").fillna(0)
            if "fraud_prediction" in df.columns:
                df["fraud_prediction"] = pd.to_numeric(df["fraud_prediction"], errors="coerce").fillna(0)

            # ==============================
            # 🚨 ALERT BANNER (NEW)
            # ==============================
            avg_prob = df[prob_col].astype(float).mean()
            if "fraud_prediction" in df.columns:
                fraud_rate = df["fraud_prediction"].astype(float).mean()
            else:
                fraud_rate = 0.0

            if avg_prob > 0.7 or fraud_rate > 0.5:
                st.error(f"🚨 High Fraud Activity! Avg Prob: {avg_prob:.2f} | Fraud Rate: {fraud_rate:.2%}")
            elif avg_prob > 0.5:
                st.warning(f"⚠️ Elevated Fraud Risk. Avg Prob: {avg_prob:.2f}")
            else:
                st.success("✅ Fraud activity is within normal range.")

            # ==============================
            # 🎯 DISTRIBUTION
            # ==============================
            st.markdown("### 📊 Prediction Confidence Distribution")

            fig = px.histogram(
                df,
                x=prob_col,
                nbins=30,
                title="Prediction Confidence Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ==============================
            # 📈 TREND
            # ==============================
            if "timestamp" in df.columns:
                st.markdown("### 📈 Prediction Trend Over Time")

                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                trend_df = df.sort_values("timestamp")

                fig = px.line(
                    trend_df,
                    x="timestamp",
                    y=prob_col,
                    title="Fraud Probability Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)

            # ==============================
            # 🔥 RISK DISTRIBUTION (NEW)
            # ==============================
            st.markdown("### 🔥 Risk Distribution")

            def risk_bucket(p):
                if p < 0.3:
                    return "Low"
                elif p < 0.7:
                    return "Medium"
                else:
                    return "High"

            df["risk_level"] = df[prob_col].apply(risk_bucket)

            st.bar_chart(df["risk_level"].value_counts())

            # ==============================
            # 🚨 HIGH RISK TABLE (NEW)
            # ==============================
            st.markdown("### 🚨 High Risk Transactions")

            high_risk = df[df[prob_col] > 0.7]

            if not high_risk.empty:
                st.dataframe(high_risk.tail(20), use_container_width=True)
            else:
                st.success("No high-risk transactions detected")

            # ==============================
            # 📋 RECENT
            # ==============================
            st.markdown("### 📋 Recent Predictions")

            st.dataframe(df.tail(25), use_container_width=True, hide_index=True)

        else:
            st.info("No monitoring records yet. Run a few predictions first.")

    except Exception as e:
        st.error(f"Monitoring failed: {e}")
