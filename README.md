# 💳 Credit Risk Prediction System (Machine Learning + Flask)

An end-to-end **Machine Learning project** that predicts whether a loan applicant is a **Good Credit** or **Bad Credit** risk using the **German Credit Dataset**.  
The project includes **data preprocessing, model training, evaluation, and deployment using Flask**.

---

## 🚀 Project Overview

Banks and financial institutions need to assess credit risk before approving loans.  
This project uses **Machine Learning** to predict creditworthiness based on applicant financial and personal attributes.

### ✅ Key Features
- Data cleaning & feature engineering
- Categorical encoding + numerical scaling
- Random Forest classification model
- Probability-based risk level & credit score
- Flask web application for real-time predictions
- End-to-end deployment ready

---

## 🧠 Machine Learning Details

- **Algorithm:** Random Forest Classifier  
- **Pipeline Used:**
  - StandardScaler (numerical features)
  - OneHotEncoder (categorical features)
- **Handling Unknown Categories:** Enabled (`handle_unknown="ignore"`)

### Model Outputs
- Prediction → Good / Bad Credit  
- Probability score  
- Risk Level (Low / Medium / High)  
- Credit Score (300–850 scale)

---

## 📂 Project Structure
```text
CREDIT-RISK-PREDICTION/
|
├── templates/
│ └── index.html      # Frontend UI
├── german_credit.csv # Dataset
├── app.py            # Flask application
├── training.py       # Model training pipeline
├── credit_risk.pkl   # Trained ML model
├── README.md         # Project documentation
```
---
## 📊 Input Features Used

| Feature | Description |
|-------|------------|
| duration | Loan duration (months) |
| credit_amount | Loan amount |
| installment_rate | Installment rate (1–4) |
| age | Applicant age |
| purpose | Loan purpose |
| existing_credits | Number of existing credits |
| other_payment_plans | Other loan plans |
| credit_history | Past repayment history |
| checking_status | Checking account balance |
| savings_status | Savings status |
| employment | Employment duration |

---

## 🎯 Target Variable

- **1 → Good Credit**
- **0 → Bad Credit**

---

## 🖥️ Web Application

The Flask app allows users to:
- Enter loan and personal details
- Get instant credit prediction
- View:
  - Credit decision
  - Probability score
  - Risk level
  - Estimated credit score

---

###  Clone the Repository
```bash
git clone https://github.com/Hemanthpolineni/CREDIT-RISK-PREDICTION.git
cd CREDIT-RISK-PREDICTIO

