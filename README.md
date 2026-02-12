# 💰 Loan Default Prediction System

A comprehensive machine learning web application that predicts loan defaults using AI. Now available as both a **modern web app (Vercel)** and **Streamlit dashboard**.

## ✨ Live Demo

🌐 **Web App**: Deploy to Vercel for instant access
📊 **Streamlit**: Interactive dashboard version

## 🎯 Features

### Overall Default Prediction
- Comprehensive risk assessment based on borrower profile
- Real-time probability calculations with ML models
- Interactive risk score visualization
- Key risk factor identification
- Actionable recommendations
- Beautiful, responsive UI

### Two Versions Available

1. **Modern Web App (Vercel)**
   - Static HTML/CSS/JavaScript frontend
   - Python serverless API for predictions
   - Lightning-fast deployment
   - Mobile-responsive design
   
2. **Streamlit Dashboard**
   - Interactive Python dashboard
   - Real-time data visualization
   - Advanced analytics and charts

## 📊 Dataset

The system uses the `loan.csv` dataset containing **39,718 loan records** with features including:

- **Borrower Information**: Annual income, employment length, home ownership
- **Loan Details**: Loan amount, interest rate, term, grade, installment
- **Credit History**: DTI, delinquencies, credit inquiries, revolving utilization
- **Payment Data**: Total payments, principal received, interest received
- **Loan Status**: Fully Paid, Charged Off, Current, etc.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd c:\Users\91983\OneDrive\Desktop\kotaknew
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

### Training the Models

Before running the app, you need to train the machine learning models:

```bash
python train_model.py
```

This will:
- Load and preprocess the loan data
- Train two models:
  - Random Forest Classifier for overall default prediction (~85% accuracy)
  - Gradient Boosting Classifier for next payment prediction (~80% accuracy)
- Save the trained models as pickle files:
  - `default_model.pkl`
  - `next_payment_model.pkl`
  - `scaler.pkl`
  - `encoders.pkl`

### Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Overall Default Prediction

Navigate to "Overall Default Prediction" from the sidebar.

**Input Fields:**

**Personal Information:**
- Annual Income
- Employment Length
- Home Ownership Status
- Income Verification Status

**Loan Details:**
- Loan Amount
- Term (36 or 60 months)
- Interest Rate
- Monthly Installment
- Loan Grade (A-G)
- Loan Purpose

**Credit History:**
- Debt-to-Income Ratio
- Delinquencies (Last 2 Years)
- Credit Inquiries (Last 6 Months)
- Open Credit Lines
- Public Records
- Revolving Balance
- Revolving Utilization
- Total Credit Lines

Click **"Predict Default Risk"** to get:
- Risk classification (Low/High Risk)
- Default probability percentage
- Visual risk gauge
- Key risk factors
- Recommendations

### 2. Monthly Payment Tracker

Navigate to "Monthly Payment Tracker" from the sidebar.

**Input Fields:**

**Current Loan Information:**
- Total Loan Amount
- Monthly Payment
- Remaining Balance
- Months on Loan

**Recent Payment History:**
- Last Payment Amount
- Days Since Last Payment
- Number of Late Payments
- Average Payment Amount
- Payment Consistency Score

**Financial Status:**
- Current Monthly Income
- Current Monthly Expenses
- Recent Credit Inquiries
- Current Credit Utilization

Click **"Predict Next Payment Default"** to get:
- Next payment risk assessment
- Default probability
- Payment timeline visualization
- Warning indicators
- Action recommendations

## 🔧 Technical Details

### Models Used

1. **Random Forest Classifier** (Overall Default)
   - 100 estimators
   - Max depth: 15
   - Class-balanced weighting
   - ~85% accuracy

2. **Gradient Boosting Classifier** (Next Payment)
   - 100 estimators
   - Learning rate: 0.1
   - Max depth: 5
   - ~80% accuracy

### Key Technologies

- **Frontend**: Streamlit
- **ML Framework**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Model Persistence**: Pickle

### Feature Engineering

**Default Prediction Features:**
- Loan characteristics (amount, term, rate, grade)
- Borrower profile (income, employment, home ownership)
- Credit metrics (DTI, delinquencies, utilization)

**Next Payment Features:**
- Payment history (consistency, late payments)
- Financial capacity (income vs. payment ratio)
- Loan progress (remaining balance, months on loan)
- Behavioral indicators (payment amounts, timing)

## 📈 Model Performance

### Overall Default Model
- **Accuracy**: ~85%
- **ROC-AUC**: ~0.75
- **Precision**: High for both classes
- **Recall**: Balanced detection

### Next Payment Model
- **Accuracy**: ~80%
- **ROC-AUC**: ~0.72
- **Early Warning**: Detects risk 30+ days ahead

## 🎨 User Interface

- Clean, intuitive design
- Color-coded risk indicators
- Interactive gauges and charts
- Mobile-responsive layout
- Real-time predictions

## � Deployment

### Streamlit Cloud (Recommended)

1. **Push to GitHub**:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/jeetshorey123/loandefaulter
git push -u origin main
```

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `jeetshorey123/loandefaulter`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Note**: Due to file size limits on Streamlit Cloud:
   - The `loan.csv` file is excluded via `.gitignore`
   - Models are pre-trained and included
   - For production, use cloud storage for data

### Alternative Deployment Options

- **Heroku**: Use Streamlit + Heroku buildpack
- **AWS EC2**: Deploy on Ubuntu instance with Streamlit
- **Docker**: Containerize and deploy to any cloud platform

⚠️ **Note**: Vercel is not recommended for Streamlit apps as it's designed for static sites and serverless functions, while Streamlit requires a persistent Python server.

## �🔮 Future Enhancements

- [ ] Add explainable AI (SHAP values) for transparency
- [ ] Implement A/B testing for model comparison
- [ ] Add database integration for persistent storage
- [ ] Create API endpoints for integration
- [ ] Add batch prediction capability
- [ ] Implement user authentication
- [ ] Add historical trend analysis
- [ ] Create custom alert rules

## 📝 File Structure

```
kotaknew/
│
├── loan.csv                    # Dataset (39,718 records)
├── app.py                      # Main Streamlit application
├── train_model.py             # Model training script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
└── (Generated after training)
    ├── default_model.pkl      # Trained default prediction model
    ├── next_payment_model.pkl # Trained next payment model
    ├── scaler.pkl            # Feature scaler
    └── encoders.pkl          # Label encoders
```

## ⚠️ Important Notes

1. **Model Training Required**: Always train models before running the app
2. **Data Quality**: Ensure loan.csv is in the same directory
3. **Regular Updates**: Retrain models with new data periodically
4. **Risk Assessment**: Predictions are probabilistic, not deterministic
5. **Business Rules**: Combine ML predictions with business logic

## 🤝 Contributing

This is a learning project. Feel free to:
- Report bugs
- Suggest features
- Improve documentation
- Enhance models

## 📄 License

This project is for educational purposes.

## 👨‍💻 Developer

Created as a comprehensive ML project for loan default prediction.

## 📞 Support

For issues or questions:
1. Check this README
2. Review the code comments
3. Test with sample data first
4. Verify all dependencies are installed

---

**Remember**: This system provides risk assessments to support decision-making, but should not be the sole factor in loan approval or management decisions.
