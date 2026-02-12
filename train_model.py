import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("LOAN DEFAULT PREDICTION - MODEL TRAINING")
print("=" * 80)

print("\n[1/7] Loading loan data...")
df = pd.read_csv('loan.csv')
print(f"✓ Loaded {len(df)} loan records with {len(df.columns)} columns")

print("\n[2/7] Preprocessing data...")

selected_columns = [
    'loan_amnt', 'funded_amnt', 'term', 'int_rate', 'installment', 
    'grade', 'sub_grade', 'emp_length', 'home_ownership', 'annual_inc',
    'verification_status', 'loan_status', 'purpose', 'dti', 
    'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
    'revol_bal', 'revol_util', 'total_acc', 'total_pymnt',
    'total_rec_prncp', 'total_rec_int', 'last_pymnt_amnt',
    'issue_d', 'application_type', 'last_credit_pull_d', 
    'total_rec_late_fee', 'last_pymnt_d'
]

df_model = df[selected_columns].copy()


df_model = df_model[df_model['loan_status'].isin(['Fully Paid', 'Charged Off'])]
df_model['default'] = (df_model['loan_status'] == 'Charged Off').astype(int)

print(f"✓ Default rate: {df_model['default'].mean()*100:.2f}%")

df_model['int_rate'] = df_model['int_rate'].str.replace('%', '').astype(float)
df_model['revol_util'] = pd.to_numeric(df_model['revol_util'].str.replace('%', ''), errors='coerce')

df_model['term'] = df_model['term'].str.extract(r'(\d+)').astype(int)


emp_length_map = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10
}
df_model['emp_length'] = df_model['emp_length'].map(emp_length_map)

le_grade = LabelEncoder()
df_model['grade_encoded'] = le_grade.fit_transform(df_model['grade'].fillna('C'))

le_home = LabelEncoder()
df_model['home_ownership_encoded'] = le_home.fit_transform(df_model['home_ownership'].fillna('RENT'))

le_verification = LabelEncoder()
df_model['verification_encoded'] = le_verification.fit_transform(df_model['verification_status'].fillna('Not Verified'))

le_purpose = LabelEncoder()
df_model['purpose_encoded'] = le_purpose.fit_transform(df_model['purpose'].fillna('other'))

le_application = LabelEncoder()
df_model['application_type_encoded'] = le_application.fit_transform(df_model['application_type'].fillna('INDIVIDUAL'))

#
df_model['issue_d'] = pd.to_datetime(df_model['issue_d'], format='%b-%y', errors='coerce')
df_model['loan_year'] = df_model['issue_d'].dt.year.fillna(2011).astype(int)


df_model['last_pymnt_d'] = pd.to_datetime(df_model['last_pymnt_d'], format='%b-%y', errors='coerce')
df_model['last_credit_pull_d'] = pd.to_datetime(df_model['last_credit_pull_d'], format='%b-%y', errors='coerce')
df_model['days_since_last_payment'] = (pd.Timestamp('2016-06-01') - df_model['last_pymnt_d']).dt.days.fillna(0)
df_model['days_since_credit_pull'] = (pd.Timestamp('2016-06-01') - df_model['last_credit_pull_d']).dt.days.fillna(0)

# Calculate payment to loan ratio - helps identify low payments on high loans
df_model['payment_to_loan_ratio'] = (df_model['last_pymnt_amnt'] / df_model['loan_amnt'] * 100).fillna(0)

# Add critical payment behavior features
df_model['total_paid_ratio'] = (df_model['total_pymnt'] / df_model['loan_amnt'] * 100).fillna(0)
df_model['principal_paid_ratio'] = (df_model['total_rec_prncp'] / df_model['loan_amnt'] * 100).fillna(0)
df_model['payment_progress'] = np.where(df_model['loan_amnt'] > 0, 
                                        df_model['total_rec_prncp'] / df_model['loan_amnt'], 0)

# Average payment amount indicator
df_model['avg_payment_ratio'] = (df_model['total_pymnt'] / df_model['installment']).fillna(0)
df_model['avg_payment_ratio'] = df_model['avg_payment_ratio'].clip(0, 100)  # Cap at reasonable range

# Payment delinquency indicators
df_model['has_late_fees'] = (df_model['total_rec_late_fee'] > 0).astype(int)
df_model['late_fee_ratio'] = (df_model['total_rec_late_fee'] / df_model['loan_amnt'] * 100).fillna(0)

numeric_columns = df_model.select_dtypes(include=[np.number]).columns
df_model[numeric_columns] = df_model[numeric_columns].fillna(df_model[numeric_columns].median())

print("✓ Data preprocessing completed")

print("\n[3/7] Preparing features for default prediction model...")

feature_columns = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade_encoded',
    'emp_length', 'home_ownership_encoded', 'annual_inc', 'verification_encoded',
    'purpose_encoded', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc',
    'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'loan_year',
    'application_type_encoded', 'total_rec_late_fee', 'last_pymnt_amnt',
    'days_since_last_payment', 'days_since_credit_pull', 'payment_to_loan_ratio',
    'total_paid_ratio', 'principal_paid_ratio', 'payment_progress',
    'avg_payment_ratio', 'has_late_fees', 'late_fee_ratio'
]

X = df_model[feature_columns]
y = df_model['default']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n[4/7] Training default prediction model...")
print("Using Random Forest Classifier with 100 trees...")


default_model = RandomForestClassifier(
    n_estimators=200,  # More trees for better pattern detection
    max_depth=20,  # Deeper trees to capture complex payment patterns
    min_samples_split=30,  # Allow finer splits
    min_samples_leaf=10,  # Smaller leaf size for better granularity
    max_features='sqrt',  # Use subset of features to reduce overfitting
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle imbalanced default rates
)

default_model.fit(X_train, y_train)


y_pred = default_model.predict(X_test)
y_pred_proba = default_model.predict_proba(X_test)[:, 1]

print("\n✓ Default Prediction Model Results:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"  ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))


feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': default_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10).to_string(index=False))

print("\n[5/7] Preparing features for next payment default model...")

np.random.seed(42)
df_payment = df_model[df_model['total_pymnt'] > 0].copy()

df_payment['remaining_balance'] = df_payment['loan_amnt'] - df_payment['total_rec_prncp']
df_payment['monthly_payment'] = df_payment['installment']
df_payment['months_on_loan'] = np.random.randint(1, df_payment['term'], size=len(df_payment))
df_payment['last_payment_amount'] = df_payment['last_pymnt_amnt']
df_payment['days_since_last_payment'] = np.random.randint(0, 60, size=len(df_payment))
df_payment['num_late_payments'] = df_payment['delinq_2yrs']
df_payment['avg_payment_amount'] = df_payment['total_pymnt'] / df_payment['months_on_loan']
df_payment['payment_consistency'] = np.random.uniform(50, 100, size=len(df_payment))
df_payment['current_income'] = df_payment['annual_inc'] / 12
df_payment['current_expenses'] = df_payment['current_income'] * df_payment['dti'] / 100

df_payment['remaining_months'] = df_payment['remaining_balance'] / df_payment['monthly_payment']
df_payment['payment_ratio'] = df_payment['last_payment_amount'] / df_payment['monthly_payment']
df_payment['income_to_payment'] = df_payment['current_income'] / df_payment['monthly_payment']
df_payment['expense_ratio'] = df_payment['current_expenses'] / df_payment['current_income']
df_payment['recent_inquiries'] = df_payment['inq_last_6mths']
df_payment['current_credit_util'] = df_payment['revol_util']

df_payment = df_payment.replace([np.inf, -np.inf], np.nan)
numeric_cols = df_payment.select_dtypes(include=[np.number]).columns
df_payment[numeric_cols] = df_payment[numeric_cols].fillna(df_payment[numeric_cols].median())

df_payment['next_payment_default'] = ((df_payment['default'] == 1) | 
                                       (df_payment['days_since_last_payment'] > 35) |
                                       (df_payment['num_late_payments'] > 1)).astype(int)

payment_features = [
    'remaining_balance', 'monthly_payment', 'months_on_loan', 'last_payment_amount',
    'days_since_last_payment', 'num_late_payments', 'avg_payment_amount',
    'payment_consistency', 'current_income', 'current_expenses', 'remaining_months',
    'payment_ratio', 'income_to_payment', 'expense_ratio', 'recent_inquiries',
    'current_credit_util'
]

X_payment = df_payment[payment_features]
y_payment = df_payment['next_payment_default']

print(f"✓ Next payment default rate: {y_payment.mean()*100:.2f}%")

# Split data
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_payment, y_payment, test_size=0.2, random_state=42, stratify=y_payment
)

print(f"✓ Training set: {len(X_train_p)} samples")
print(f"✓ Test set: {len(X_test_p)} samples")

print("\n[6/7] Training next payment default model...")
print("Using Gradient Boosting Classifier...")

next_payment_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    subsample=0.8
)

next_payment_model.fit(X_train_p, y_train_p)

y_pred_p = next_payment_model.predict(X_test_p)
y_pred_proba_p = next_payment_model.predict_proba(X_test_p)[:, 1]

print("\n✓ Next Payment Default Model Results:")
print(f"  Accuracy: {accuracy_score(y_test_p, y_pred_p)*100:.2f}%")
print(f"  ROC-AUC Score: {roc_auc_score(y_test_p, y_pred_proba_p):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_p, y_pred_p, target_names=['No Default', 'Default']))

payment_feature_importance = pd.DataFrame({
    'feature': payment_features,
    'importance': next_payment_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features for Next Payment:")
print(payment_feature_importance.head(10).to_string(index=False))

print("\n[7/7] Saving models...")

with open('default_model.pkl', 'wb') as f:
    pickle.dump(default_model, f)
print("✓ Saved: default_model.pkl")

with open('next_payment_model.pkl', 'wb') as f:
    pickle.dump(next_payment_model, f)
print("✓ Saved: next_payment_model.pkl")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved: scaler.pkl")

# Save encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump({
        'grade': le_grade,
        'home_ownership': le_home,
        'verification': le_verification,
        'purpose': le_purpose
    }, f)
print("✓ Saved: encoders.pkl")

print("\n" + "=" * 80)
print("✓ MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nNext steps:")
print("1. Run the Streamlit app: streamlit run app.py")
print("2. Test predictions with sample borrower data")
print("3. Monitor model performance and retrain as needed")
print("=" * 80)
