import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Loan Default Prediction System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 40px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 20px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .safe {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .risky {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        # Handle NumPy version compatibility issues
        import numpy as np
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        
        with open('default_model.pkl', 'rb') as f:
            default_model = pickle.load(f)
        with open('next_payment_model.pkl', 'rb') as f:
            next_payment_model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        np.load = np_load_old
        return default_model, next_payment_model, scaler
    except (FileNotFoundError, ValueError) as e:
        if "BitGenerator" in str(e):
            st.warning("⚠️ Model compatibility issue detected. Please retrain the models by running: `python train_model.py`")
        return None, None, None

st.markdown('<div class="main-header">💰 Loan Default Prediction System</div>', unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a prediction type:", 
                        ["Overall Default Prediction"])

if page == "Overall Default Prediction":
    st.markdown('<div class="sub-header">📊 Overall Loan Default Risk Assessment</div>', unsafe_allow_html=True)
    st.write("Enter borrower details to predict the likelihood of loan default.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Personal Information")
        annual_inc = st.number_input("Annual Income ($)", min_value=1000, max_value=10000000, value=50000, step=1000)
        dti = st.selectbox("Debt-to-Income Ratio (%)", 
                          ["0-5%", "5-10%", "10-15%", "15-20%", "20-25%", "25-30%", "30-35%", "35-40%", "40%+"],
                          index=3)
    
    with col2:
        st.subheader("Loan Details")
        loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=40000, value=10000, step=500)
        term = st.selectbox("Term", [" 36 months", " 60 months"])
        int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.0, step=0.1)
        installment = st.number_input("Monthly Installment ($)", min_value=50, max_value=2000, value=300, step=10)
        loan_issue_date = st.date_input("Loan Issue Date", value=pd.Timestamp('2023-01-01'), min_value=pd.Timestamp('2007-01-01'), max_value=pd.Timestamp('2026-01-12'))
    
    with col3:
        st.subheader("Credit Information")
        open_acc = st.number_input("Open Credit Lines", min_value=0, max_value=50, value=10)
        total_acc = st.number_input("Total Credit Lines", min_value=0, max_value=100, value=20)
        revol_bal = st.number_input("Revolving Balance ($)", min_value=0, max_value=500000, value=10000, step=500)
        revol_util = st.slider("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    
    st.markdown("---")
    col4, col5 = st.columns(2)
    
    with col4:
        st.subheader("Repayment History")
        months_since_issue = ((pd.Timestamp('2026-01-12') - pd.Timestamp(loan_issue_date)).days // 30)
        total_paid_so_far = st.number_input("Total Amount Paid So Far ($)", min_value=0, max_value=loan_amnt*2, value=min(int(installment * min(months_since_issue, 12)), loan_amnt*2), step=100)
        num_payments_made = st.number_input("Number of Payments Made", min_value=0, max_value=60, value=min(months_since_issue, 12))
        total_rec_late_fee = st.number_input("Total Late Fees Received ($)", min_value=0.0, max_value=10000.0, value=0.0, step=10.0)
    
    with col5:
        st.subheader("Payment Information")
        last_pymnt_amnt = st.number_input("Last Payment Amount ($)", min_value=0, max_value=50000, value=300, step=50)
        last_pymnt_date = st.date_input("Last Payment Date", value=pd.Timestamp('2024-01-01'))
        last_credit_pull_date = st.date_input("Last Credit Pull Date", value=pd.Timestamp('2024-06-01'))
        
        st.info("💡 **Tip:** If making regular payments, set 'Last Payment Date' to within the last 30-60 days for accurate risk assessment.")
    
    if st.button("🔮 Predict Default Risk", key="predict_default"):
        default_model, next_payment_model, scaler = load_models()
        
        if default_model is None:
            st.error("⚠️ Models not found! Please train the models first by running train_model.py")

            
        else:
            dti_map = {
                "0-5%": 2.5, "5-10%": 7.5, "10-15%": 12.5, "15-20%": 17.5,
                "20-25%": 22.5, "25-30%": 27.5, "30-35%": 32.5, "35-40%": 37.5, "40%+": 45.0
            }
            dti_value = dti_map[dti]
            
            from datetime import datetime
            reference_date = pd.Timestamp('2026-01-12')
            days_since_last_payment = (reference_date - pd.Timestamp(last_pymnt_date)).days
            days_since_credit_pull = (reference_date - pd.Timestamp(last_credit_pull_date)).days
            days_since_loan_issue = (reference_date - pd.Timestamp(loan_issue_date)).days
            
            # Calculate payment to loan ratio - low payment on high loan = high risk
            payment_to_loan_ratio = (last_pymnt_amnt / loan_amnt * 100) if loan_amnt > 0 else 0
            
            # Calculate repayment metrics
            expected_payments = max(1, days_since_loan_issue // 30)
            payment_compliance_rate = (num_payments_made / expected_payments * 100) if expected_payments > 0 else 100
            total_paid_ratio = (total_paid_so_far / loan_amnt * 100) if loan_amnt > 0 else 0
            avg_monthly_payment = total_paid_so_far / num_payments_made if num_payments_made > 0 else 0
            
            loan_year = pd.Timestamp(loan_issue_date).year
            
            # Set default values for removed fields
            emp_length = 5  # Default to 5 years
            grade = 'C'  # Default to grade C
            purpose = 'debt_consolidation'  # Default purpose
            home_ownership = 'RENT'  # Default home ownership
            verification_status = 'Not Verified'  # Default verification
            application_type = 'INDIVIDUAL'  # Default application type
            delinq_2yrs = 0  # Default no delinquencies
            inq_last_6mths = 0  # Default no inquiries
            pub_rec = 0  # Default no public records
            
            features = pd.DataFrame({
                'loan_amnt': [loan_amnt],
                'term': [36 if term == " 36 months" else 60],
                'int_rate': [int_rate],
                'installment': [installment],
                'grade_encoded': [ord(grade) - ord('A')],
                'emp_length': [emp_length],
                'home_ownership_encoded': [0 if home_ownership == "RENT" else 1 if home_ownership == "MORTGAGE" else 2],
                'annual_inc': [annual_inc],
                'verification_encoded': [0 if verification_status == "Not Verified" else 1],
                'purpose_encoded': [hash(purpose) % 100],
                'dti': [dti_value],
                'delinq_2yrs': [delinq_2yrs],
                'inq_last_6mths': [inq_last_6mths],
                'open_acc': [open_acc],
                'pub_rec': [pub_rec],
                'revol_bal': [revol_bal],
                'revol_util': [revol_util],
                'total_acc': [total_acc],
                'loan_year': [loan_year],
                'application_type_encoded': [0 if application_type == "INDIVIDUAL" else 1],
                'total_rec_late_fee': [total_rec_late_fee],
                'last_pymnt_amnt': [last_pymnt_amnt],
                'days_since_last_payment': [days_since_last_payment],
                'days_since_credit_pull': [days_since_credit_pull],
                'payment_to_loan_ratio': [payment_to_loan_ratio],
                'total_paid_ratio': [total_paid_ratio],
                'principal_paid_ratio': [total_paid_ratio * 0.85],  # Approximate (85% of payment goes to principal)
                'payment_progress': [total_paid_ratio / 100],
                'avg_payment_ratio': [avg_monthly_payment / installment if installment > 0 else 0],
                'has_late_fees': [1 if total_rec_late_fee > 0 else 0],
                'late_fee_ratio': [(total_rec_late_fee / loan_amnt * 100) if loan_amnt > 0 else 0]
            })
            
            prediction_proba = default_model.predict_proba(features)[0]
            prediction = default_model.predict(features)[0]
            
            # === FORMULA-BASED INTELLIGENT RISK SCORING ===
            # Start with ML model's base prediction
            base_ml_risk = prediction_proba[1] * 100
            
            # Calculate expected vs actual payment metrics
            expected_total_payment = expected_payments * installment
            actual_payment_ratio = (total_paid_so_far / expected_total_payment) if expected_total_payment > 0 else 1.0
            
            # === COMPONENT 1: Payment Compliance Score (0-100, higher is better) ===
            compliance_score = payment_compliance_rate
            
            # === COMPONENT 2: Payment Amount Score (0-100, higher is better) ===
            # Compare average payment to required installment
            payment_amount_score = min(100, (avg_monthly_payment / installment * 100)) if installment > 0 else 100
            
            # === COMPONENT 3: Payment Progress Score (0-100, higher is better) ===
            # Are they ahead, on-track, or behind expected payments?
            if actual_payment_ratio >= 1.0:  # Ahead of schedule
                progress_score = 100
            elif actual_payment_ratio >= 0.9:  # Within 10% of expected
                progress_score = 90
            elif actual_payment_ratio >= 0.8:  # Within 20% of expected
                progress_score = 80
            elif actual_payment_ratio >= 0.7:  # Within 30% of expected
                progress_score = 70
            elif actual_payment_ratio >= 0.5:  # Paid at least 50% of expected
                progress_score = 50
            else:  # Significantly behind
                progress_score = max(0, actual_payment_ratio * 100)
            
            # === COMPONENT 4: Recency Score (0-100, higher is better) ===
            # Recent payments indicate active engagement
            if days_since_last_payment <= 30:
                recency_score = 100
            elif days_since_last_payment <= 60:
                recency_score = 80
            elif days_since_last_payment <= 90:
                recency_score = 60
            elif days_since_last_payment <= 180:
                recency_score = 40
            elif days_since_last_payment <= 365:
                recency_score = 20
            else:
                recency_score = 0
            
            # === COMPONENT 5: Late Fee Penalty (0-100, lower is worse) ===
            late_fee_penalty = 0 if total_rec_late_fee == 0 else min(30, (total_rec_late_fee / loan_amnt * 100) * 10)
            
            # === FORMULA: Calculate Overall Payment Health Score (0-100) ===
            # Weighted combination of all components
            payment_health_score = (
                compliance_score * 0.35 +       # 35% weight - most important
                progress_score * 0.30 +         # 30% weight - are they on track?
                payment_amount_score * 0.20 +   # 20% weight - paying enough?
                recency_score * 0.15            # 15% weight - recent activity
            ) - late_fee_penalty
            
            payment_health_score = max(0, min(100, payment_health_score))
            
            # === RISK CALCULATION FORMULA ===
            # Blend ML model prediction with payment health score
            # Good payment health should significantly reduce ML risk
            
            if payment_health_score >= 90:
                # Excellent payment behavior - override ML with low risk
                risk_score = min(base_ml_risk * 0.2, 20)  # Max 20% risk
            elif payment_health_score >= 80:
                # Very good payment behavior
                risk_score = min(base_ml_risk * 0.3, 30)  # Max 30% risk
            elif payment_health_score >= 70:
                # Good payment behavior
                risk_score = min(base_ml_risk * 0.5, 40)  # Max 40% risk
            elif payment_health_score >= 60:
                # Above average payment behavior
                risk_score = base_ml_risk * 0.7  # 30% reduction
            elif payment_health_score >= 50:
                # Average payment behavior
                risk_score = base_ml_risk * 0.85  # 15% reduction
            elif payment_health_score >= 40:
                # Below average - slight reduction
                risk_score = base_ml_risk * 0.95  # 5% reduction
            else:
                # Poor payment behavior - increase risk
                risk_score = min(100, base_ml_risk * 1.2)  # 20% increase
            
            # === CRITICAL OVERRIDE CONDITIONS ===
            # Absolute red flags that override everything
            
            # No payment for over 1 year
            if days_since_last_payment > 365:
                risk_score = max(risk_score, 90)
            
            # Less than 10% paid after 1+ year
            if days_since_loan_issue > 365 and total_paid_ratio < 10:
                risk_score = max(risk_score, 85)
            
            # Payment compliance below 40% (missing most payments)
            if payment_compliance_rate < 40 and expected_payments > 6:
                risk_score = max(risk_score, 80)
            
            # Very high late fees (>5% of loan)
            if total_rec_late_fee > loan_amnt * 0.05:
                risk_score = min(100, risk_score + 15)
            
            # Income factor
            if annual_inc < 20000:
                risk_score = min(100, risk_score + 10)
            
            # Update prediction based on adjusted risk score
            prediction = 1 if risk_score >= 50 else 0
            prediction_proba = [1 - (risk_score/100), risk_score/100]
            
            st.markdown("---")
            
            # Show payment status summary
            st.subheader("📋 Payment Status Summary")
            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
            
            with col_sum1:
                st.metric("Payment Compliance", f"{payment_compliance_rate:.0f}%", 
                         "✅ Good" if payment_compliance_rate >= 80 else "⚠️ Poor")
            with col_sum2:
                st.metric("Amount Paid", f"${total_paid_so_far:,}", 
                         f"{total_paid_ratio:.1f}% of loan")
            with col_sum3:
                st.metric("Avg Payment", f"${avg_monthly_payment:.0f}", 
                         "✅ On track" if avg_monthly_payment >= installment * 0.9 else "⚠️ Below required")
            with col_sum4:
                st.metric("Days Since Last Payment", f"{days_since_last_payment}", 
                         "✅ Recent" if days_since_last_payment <= 60 else "⚠️ Overdue" if days_since_last_payment <= 180 else "🚨 Critical")
            
            st.markdown("---")
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if prediction == 1:
                    st.markdown(f'<div class="prediction-box risky"><h3>⚠️ HIGH RISK</h3><p>Default Probability: <b>{prediction_proba[1]*100:.2f}%</b></p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box safe"><h3>✅ LOW RISK</h3><p>Default Probability: <b>{prediction_proba[1]*100:.2f}%</b></p></div>', unsafe_allow_html=True)
            
            with col_res2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction_proba[1]*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Default Risk Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50}}))
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📈 Key Risk Factors")
            risk_factors = []
            
            # Critical risk factors
            if total_paid_ratio < 10 and days_since_loan_issue > 365:
                risk_factors.append(f"CRITICAL: Only {total_paid_ratio:.2f}% of loan repaid (${total_paid_so_far:,.0f} of ${loan_amnt:,.0f}) after {days_since_loan_issue} days")
            if payment_compliance_rate < 50:
                risk_factors.append(f"CRITICAL: Only {num_payments_made} payments made out of {expected_payments} expected ({payment_compliance_rate:.0f}% compliance)")
            if payment_to_loan_ratio < 1:
                risk_factors.append(f"CRITICAL: Last payment (${last_pymnt_amnt}) is only {payment_to_loan_ratio:.2f}% of loan amount (${loan_amnt})")
            if days_since_last_payment > 365:
                risk_factors.append(f"CRITICAL: No payment for {days_since_last_payment} days ({days_since_last_payment/365:.1f} years)")
            elif days_since_last_payment > 90:
                risk_factors.append(f"HIGH RISK: Last payment was {days_since_last_payment} days ago")
            
            # Payment tracking risks
            if avg_monthly_payment < installment * 0.5 and num_payments_made > 0:
                risk_factors.append(f"Average payment (${avg_monthly_payment:.0f}) is much less than required installment (${installment})")
            
            # High risk factors
            if last_pymnt_amnt < installment * 0.5:
                risk_factors.append(f"Last payment (${last_pymnt_amnt}) is much less than monthly installment (${installment})")
            if dti_value > 30:
                risk_factors.append(f"High Debt-to-Income Ratio: {dti}")
            if revol_util > 80:
                risk_factors.append(f"High Credit Utilization: {revol_util:.1f}%")
            
            # Moderate risk factors
            if int_rate > 15:
                risk_factors.append(f"High Interest Rate: {int_rate:.1f}%")
            if annual_inc < 30000:
                risk_factors.append(f"Low Annual Income: ${annual_inc:,}")
            if total_rec_late_fee > 0:
                risk_factors.append(f"Late Fees Accumulated: ${total_rec_late_fee:.2f}")
            
            if risk_factors:
                for factor in risk_factors:
                    st.warning(f"⚠️ {factor}")
            else:
                st.success("✅ No major risk factors identified")
