from http.server import BaseHTTPRequestHandler
import json
import numpy as np
from datetime import datetime

class handler(BaseHTTPRequestHandler):
    
    def do_POST(self):
        # Read request body
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        try:
            # Process input data (lightweight rule-based prediction)
            result = self.make_prediction(data)
            
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
    
    def make_prediction(self, data):
        """
        Lightweight rule-based prediction algorithm.
        For full ML model, upgrade to Vercel Pro or use external ML API.
        This demo uses risk scoring based on financial indicators.
        """
        # Extract data
        annual_inc = float(data['annual_inc'])
        dti = float(data['dti'])
        loan_amnt = float(data['loan_amnt'])
        term = int(data['term'])
        int_rate = float(data['int_rate'])
        installment = float(data['installment'])
        loan_issue_date = data['loan_issue_date']
        total_paid_so_far = float(data['total_paid_so_far'])
        num_payments_made = int(data['num_payments_made'])
        total_rec_late_fee = float(data['total_rec_late_fee'])
        last_pymnt_amnt = float(data['last_pymnt_amnt'])
        last_pymnt_date = data['last_pymnt_date']
        revol_util = float(data['revol_util'])
        
        # Calculate date-based features
        from datetime import datetime
        reference_date = datetime(2026, 2, 12)
        loan_issue_dt = datetime.fromisoformat(loan_issue_date)
        last_pymnt_dt = datetime.fromisoformat(last_pymnt_date)
        
        days_since_last_payment = (reference_date - last_pymnt_dt).days
        days_since_loan_issue = (reference_date - loan_issue_dt).days
        
        # Calculate derived metrics
        expected_payments = max(1, days_since_loan_issue // 30)
        payment_compliance_rate = (num_payments_made / expected_payments * 100) if expected_payments > 0 else 100
        total_paid_ratio = (total_paid_so_far / loan_amnt * 100) if loan_amnt > 0 else 0
        avg_monthly_payment = total_paid_so_far / num_payments_made if num_payments_made > 0 else 0
        payment_to_loan_ratio = (last_pymnt_amnt / loan_amnt * 100) if loan_amnt > 0 else 0
        
        # Risk scoring algorithm (0-100)
        risk_score = 0
        
        # Factor 1: Payment history (40 points)
        if payment_compliance_rate < 50:
            risk_score += 40
        elif payment_compliance_rate < 80:
            risk_score += 25
        elif payment_compliance_rate < 95:
            risk_score += 10
        else:
            risk_score += 0
            
        # Factor 2: Days since last payment (25 points)
        if days_since_last_payment > 90:
            risk_score += 25
        elif days_since_last_payment > 60:
            risk_score += 18
        elif days_since_last_payment > 30:
            risk_score += 10
        else:
            risk_score += 0
            
        # Factor 3: Late fees (15 points)
        if total_rec_late_fee > 100:
            risk_score += 15
        elif total_rec_late_fee > 50:
            risk_score += 10
        elif total_rec_late_fee > 0:
            risk_score += 5
            
        # Factor 4: Interest rate (10 points) - high rate = risky borrower
        if int_rate > 20:
            risk_score += 10
        elif int_rate > 15:
            risk_score += 6
        elif int_rate > 12:
            risk_score += 3
            
        # Factor 5: DTI ratio (10 points)
        if dti > 35:
            risk_score += 10
        elif dti > 25:
            risk_score += 6
        elif dti > 20:
            risk_score += 3
            
        # Factor 6: Revolving utilization (5 points)
        if revol_util > 80:
            risk_score += 5
        elif revol_util > 60:
            risk_score += 3
            
        # Adjust based on payment amount vs expected
        if avg_monthly_payment < installment * 0.5:
            risk_score += 5
        
        # Normalize to percentage
        default_probability = min(100, max(0, risk_score))
        
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
