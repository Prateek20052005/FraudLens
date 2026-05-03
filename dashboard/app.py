# dashboard/app.py

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import time
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FraudLens Dashboard",
    page_icon="🔍",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

st.title("🔍 FraudLens — Fraud Detection Dashboard")

# --- SIDEBAR ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Live Monitor", "Analyze Transaction", "Model Stats"])

# --- PAGE 1: LIVE MONITOR ---
if page == "Live Monitor":
    st.header("📊 Live Transaction Monitor")
    st.write("Simulating real-time transactions and scoring them for fraud.")
    
    # Load sample transactions
    sample_df = pd.read_parquet("/Users/prateek/Desktop/FraudLens/data/sample_transactions.parquet")
    # Controls
    col1, col2 = st.columns(2)
    with col1:
        n_transactions = st.slider("Number of transactions to simulate", 5, 50, 10)
    with col2:
        speed = st.slider("Simulation speed (seconds between transactions)", 0.1, 2.0, 0.5)
    
    if st.button("▶ Start Simulation", type="primary"):
        # Create containers for live updates
        metrics_container = st.container()
        table_container = st.container()
        chart_container = st.container()
        
        results = []
        progress_bar = st.progress(0)
        
        for i in range(n_transactions):
            # Pick a random transaction from sample data
            row = sample_df.iloc[np.random.randint(0, len(sample_df))]
            
            # Build the request payload with available fields
            payload = {
                "TransactionAmt": float(row.get("TransactionAmt", 50)),
                "card1": int(row.get("card1", 1000)),
                "C1": float(row.get("C1", 0)),
                "C2": float(row.get("C2", 0)),
                "C3": float(row.get("C3", 0)),
                "C4": float(row.get("C4", 0)),
                "C5": float(row.get("C5", 0)),
                "C6": float(row.get("C6", 0)),
                "C7": float(row.get("C7", 0)),
                "C8": float(row.get("C8", 0)),
                "C9": float(row.get("C9", 0)),
                "C10": float(row.get("C10", 0)),
                "C11": float(row.get("C11", 0)),
                "C12": float(row.get("C12", 0)),
                "C13": float(row.get("C13", 0)),
                "C14": float(row.get("C14", 0)),
                "TransactionDT": int(row.get("TransactionDT", 86400)),
            }
            
            # Add optional fields if they exist and aren't NaN
            optional_fields = {
                "card2": "card2", "card3": "card3", "card5": "card5",
                "addr1": "addr1", "addr2": "addr2",
                "dist1": "dist1", "dist2": "dist2",
                "D1": "D1", "D2": "D2", "D3": "D3", "D4": "D4",
                "D5": "D5", "D6": "D6", "D7": "D7", "D8": "D8",
                "D9": "D9", "D10": "D10", "D11": "D11", "D15": "D15",
            }
            
            for api_field, df_field in optional_fields.items():
                if df_field in row.index and pd.notna(row[df_field]):
                    payload[api_field] = float(row[df_field])
            
            # Call the API
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                result = response.json()
                
                results.append({
                    "Transaction #": i + 1,
                    "Amount": f"${payload['TransactionAmt']:.2f}",
                    "Fraud Prob": f"{result['fraud_probability']*100:.1f}%",
                    "Risk Level": result['risk_level'],
                    "Top Reason": result['reasons'][0]['feature'],
                    "raw_prob": result['fraud_probability']
                })
            except Exception as e:
                results.append({
                    "Transaction #": i + 1,
                    "Amount": f"${payload['TransactionAmt']:.2f}",
                    "Fraud Prob": "ERROR",
                    "Risk Level": "ERROR",
                    "Top Reason": str(e),
                    "raw_prob": 0
                })
            
            # Update progress
            progress_bar.progress((i + 1) / n_transactions)
            
            # Update metrics
            with metrics_container:
                m1, m2, m3, m4 = st.columns(4)
                total = len(results)
                high_risk = sum(1 for r in results if r['Risk Level'] == 'HIGH')
                medium_risk = sum(1 for r in results if r['Risk Level'] == 'MEDIUM')
                avg_prob = np.mean([r['raw_prob'] for r in results])
                
                m1.metric("Transactions Processed", total)
                m2.metric("High Risk", high_risk)
                m3.metric("Medium Risk", medium_risk)
                m4.metric("Avg Fraud Probability", f"{avg_prob*100:.1f}%")
            
            # Update table
            with table_container:
                display_df = pd.DataFrame(results)
                
                def color_risk(val):
                    if val == "HIGH": return "background-color: #ff4444; color: white"
                    elif val == "MEDIUM": return "background-color: #ffaa00; color: white"
                    elif val == "LOW": return "background-color: #44bb44; color: white"
                    return ""
                
                st.dataframe(
                    display_df[["Transaction #", "Amount", "Fraud Prob", "Risk Level", "Top Reason"]],
                    use_container_width=True,
                    hide_index=True
                )
            
            time.sleep(speed)
        
        # Final chart
        with chart_container:
            st.subheader("Fraud Probability Distribution")
            fig = go.Figure()
            
            probs = [r['raw_prob'] for r in results]
            colors = ['#ff4444' if p > 0.7 else '#ffaa00' if p > 0.3 else '#44bb44' for p in probs]
            
            fig.add_trace(go.Bar(
                x=list(range(1, len(probs) + 1)),
                y=probs,
                marker_color=colors,
                text=[f"{p*100:.1f}%" for p in probs],
                textposition='auto'
            ))
            
            fig.update_layout(
                xaxis_title="Transaction #",
                yaxis_title="Fraud Probability",
                yaxis_range=[0, 1],
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: ANALYZE TRANSACTION ---
elif page == "Analyze Transaction":
    st.header("🔎 Analyze a Single Transaction")
    st.write("Enter transaction details and get a fraud risk assessment with explanation.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Transaction Info")
        amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=75.0)
        transaction_dt = st.number_input("Transaction Time (seconds)", min_value=0, value=86400)
        product_cd = st.selectbox("Product Category", ["W", "H", "C", "R", "S"])
        dist1 = st.number_input("Distance", min_value=0.0, value=19.0)
    
    with col2:
        st.subheader("Card Info")
        card1 = st.number_input("Card1 ID", min_value=0, value=13926)
        card4 = st.selectbox("Card Network", ["visa", "mastercard", "discover", "american express"])
        card6 = st.selectbox("Card Type", ["debit", "credit"])
    
    with col3:
        st.subheader("Identity Info")
        email = st.text_input("Purchaser Email Domain", value="gmail.com")
        device = st.selectbox("Device Type", ["desktop", "mobile", None])
        browser = st.text_input("Browser", value="chrome 65.0")
    
    # C features in an expander
    with st.expander("Advanced: C-features (counting)"):
        c_cols = st.columns(7)
        c_values = {}
        for i, c in enumerate([f"C{j}" for j in range(1, 15)]):
            with c_cols[i % 7]:
                c_values[c] = st.number_input(c, min_value=0.0, value=1.0, key=c)
    
    if st.button("🔍 Analyze Transaction", type="primary"):
        payload = {
            "TransactionAmt": amount,
            "card1": card1,
            "card4": card4,
            "card6": card6,
            "TransactionDT": transaction_dt,
            "ProductCD": product_cd,
            "dist1": dist1,
            "P_emaildomain": email,
            "DeviceType": device,
            "id_31": browser,
            **c_values
        }
        
        with st.spinner("Analyzing transaction..."):
            response = requests.post(f"{API_URL}/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Display result
            prob = result['fraud_probability']
            risk = result['risk_level']
            
            # Big metric display
            r1, r2 = st.columns(2)
            with r1:
                if risk == "HIGH":
                    st.error(f"🚨 FRAUD PROBABILITY: {prob*100:.1f}%")
                elif risk == "MEDIUM":
                    st.warning(f"⚠️ FRAUD PROBABILITY: {prob*100:.1f}%")
                else:
                    st.success(f"✅ FRAUD PROBABILITY: {prob*100:.1f}%")
            
            with r2:
                st.metric("Risk Level", risk)
            
            # SHAP waterfall chart
            st.subheader("Why this prediction?")
            
            reasons = result['reasons']
            features = [r['feature'] for r in reasons]
            impacts = [r['impact'] for r in reasons]
            colors = ['#ff4444' if i > 0 else '#4488ff' for i in impacts]
            
            fig = go.Figure(go.Bar(
                x=impacts,
                y=features,
                orientation='h',
                marker_color=colors,
                text=[f"{i:+.3f}" for i in impacts],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Top Feature Contributions to Fraud Score",
                xaxis_title="SHAP Impact (→ fraud | ← legitimate)",
                yaxis=dict(autorange="reversed"),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed reasons table
            st.subheader("Detailed Breakdown")
            reasons_df = pd.DataFrame(reasons)
            st.dataframe(reasons_df, use_container_width=True, hide_index=True)
        else:
            st.error(f"API Error: {response.text}")

# --- PAGE 3: MODEL STATS ---
elif page == "Model Stats":
    st.header("📈 Model Performance & Statistics")
    
    # Fetch stats from API
    try:
        response = requests.get(f"{API_URL}/stats")
        stats = response.json()
        
        if "message" in stats:
            st.info("No predictions yet. Go to Live Monitor to generate some predictions first.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Predictions", stats['total_predictions'])
            col2.metric("Avg Fraud Probability", f"{stats['avg_fraud_probability']*100:.1f}%")
            col3.metric("Max Fraud Probability", f"{stats['max_fraud_probability']*100:.1f}%")
            col4.metric("Avg Transaction Amount", f"${stats['avg_transaction_amount']:.2f}")
            
            # Risk distribution pie chart
            if stats.get('risk_distribution'):
                st.subheader("Risk Level Distribution")
                risk_data = stats['risk_distribution']
                
                fig = go.Figure(data=[go.Pie(
                    labels=list(risk_data.keys()),
                    values=list(risk_data.values()),
                    marker_colors=['#ff4444' if k == 'HIGH' else '#ffaa00' if k == 'MEDIUM' else '#44bb44' 
                                   for k in risk_data.keys()],
                    hole=0.4
                )])
                
                fig.update_layout(title="Predictions by Risk Level", height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure the FastAPI server is running on port 8000.")
    
    # Model info
    st.subheader("Model Information")
    st.write("**Model:** LightGBM v5 (gradient boosted trees)")
    st.write("**Features:** 290 engineered features")
    st.write("**Training Data:** IEEE-CIS Fraud Detection (590K transactions)")
    st.write("**Validation ROC-AUC:** 0.9280")
    st.write("**Validation PR-AUC:** 0.5997")
    st.write("**Explainability:** SHAP TreeExplainer")

