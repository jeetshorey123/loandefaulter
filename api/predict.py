from http.server import BaseHTTPRequestHandler
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class handler(BaseHTTPRequestHandler):
    
    def do_POST(self):
        # Read request body
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        try:
            # Load models
            root_dir = os.path.dirname(os.path.dirname(__file__))
            
            # Handle NumPy version compatibility
            np_load_old = np.load
            np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
            
            with open(os.path.join(root_dir, 'default_model.pkl'), 'rb') as f:
                default_model = pickle.load(f)
            with open(os.path.join(root_dir, 'scaler.pkl'), 'rb') as f:
                scaler = pickle.load(f)
                
            np.load = np_load_old
            
            # Process input data
            result = self.make_prediction(data, default_model, scaler)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            # Send error response
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            error_response = {'error': str(e)}
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_OPTIONS(self):
        # Handle CORS preflight
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def make_prediction(self, data, model, scaler):
        # Extract data
        annual_inc = float(data['annual_inc'])
        dti = float(data['dti'])
        loan_amnt = float(data['loan_amnt'])
        term = int(data['term'])
        int_rate = float(data['int_rate'])
        installment = float(data['installment'])
        loan_issue_date = data['loan_issue_date']
        open_acc = int(data['open_acc'])
        total_acc = int(data['total_acc'])
        revol_bal = float(data['revol_bal'])
        revol_util = float(data['revol_util'])
        total_paid_so_far = float(data['total_paid_so_far'])
        num_payments_made = int(data['num_payments_made'])
        total_rec_late_fee = float(data['total_rec_late_fee'])
        last_pymnt_amnt = float(data['last_pymnt_amnt'])
        last_pymnt_date = data['last_pymnt_date']
        last_credit_pull_date = data['last_credit_pull_date']
        
        # Calculate date-based features
        reference_date = pd.Timestamp('2026-02-12')
        loan_issue_pd = pd.Timestamp(loan_issue_date)
        last_pymnt_pd = pd.Timestamp(last_pymnt_date)
        last_credit_pull_pd = pd.Timestamp(last_credit_pull_date)
        
        days_since_last_payment = (reference_date - last_pymnt_pd).days
        days_since_credit_pull = (reference_date - last_credit_pull_pd).days
        days_since_loan_issue = (reference_date - loan_issue_pd).days
        
        # Calculate derived features
        payment_to_loan_ratio = (last_pymnt_amnt / loan_amnt * 100) if loan_amnt > 0 else 0
        expected_payments = max(1, days_since_loan_issue // 30)
        payment_compliance_rate = (num_payments_made / expected_payments * 100) if expected_payments > 0 else 100
        total_paid_ratio = (total_paid_so_far / loan_amnt * 100) if loan_amnt > 0 else 0
        avg_monthly_payment = total_paid_so_far / num_payments_made if num_payments_made > 0 else 0
        
        loan_year = loan_issue_pd.year
        last_payment_month = last_pymnt_pd.month
        credit_pull_month = last_credit_pull_pd.month
        
        # Calculate engineered features (as in training)
        total_pymnt = total_paid_so_far
        total_rec_prncp = total_paid_so_far * 0.7  # Approximate
        total_rec_int = total_paid_so_far * 0.3   # Approximate
        funded_amnt = loan_amnt
        
        principal_paid_ratio = (total_rec_prncp / loan_amnt * 100) if loan_amnt > 0 else 0
        payment_progress = (num_payments_made / term * 100) if term > 0 else 0
        total_paid_ratio_calc = (total_pymnt / funded_amnt * 100) if funded_amnt > 0 else 0
        avg_payment_ratio = (avg_monthly_payment / installment * 100) if installment > 0 else 0
        payment_to_loan_ratio_calc = (total_pymnt / loan_amnt * 100) if loan_amnt > 0 else 0
        
        # Prepare feature vector (matching training features)
        features = np.array([[
            principal_paid_ratio,
            payment_progress,
            total_paid_ratio_calc,
            avg_payment_ratio,
            payment_to_loan_ratio_calc,
            last_pymnt_amnt,
            days_since_last_payment,
            term,
            int_rate,
            0,  # grade_encoded (placeholder)
            dti,
            revol_util,
            annual_inc,
            installment,
            loan_amnt,
            days_since_credit_pull,
            total_acc,
            open_acc,
            revol_bal,
            total_rec_late_fee
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction_proba = model.predict_proba(features_scaled)[0]
        default_probability = prediction_proba[1] * 100
        
        # Prepare response
        result = {
            'default_probability': round(default_probability, 2),
            'prediction': 'High Risk' if default_probability >= 50 else 'Low Risk',
            'metrics': {
                'payment_to_loan_ratio': round(payment_to_loan_ratio, 2),
                'total_paid_ratio': round(total_paid_ratio, 2),
                'payment_compliance_rate': round(payment_compliance_rate, 2),
                'avg_monthly_payment': round(avg_monthly_payment, 2),
                'days_since_last_payment': days_since_last_payment,
                'expected_payments': expected_payments,
                'actual_payments': num_payments_made
            }
        }
        
        return result
