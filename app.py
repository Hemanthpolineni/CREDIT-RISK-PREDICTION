import joblib
import numpy as np
import pandas as pd
from flask import Flask , render_template,request

app = Flask(__name__)

model = joblib.load("credit_risk.pkl")

@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict',methods = ['POST'])
def predict():

    data = request.form

    input_data = pd.DataFrame([{

            'duration' : int(data['duration']),
            'credit_amount' : int(data['credit_amount']),
            'installment_rate' : int(data['installment_rate']),
            'age' : int(data['age']),
            'purpose' : (data['purpose']),
            'existing_credits' : int(data['existing_credits']),
            'other_payment_plans' : data['other_payment_plans'],
            'credit_history' : data['credit_history'],
            'checking_status' : data['checking_status'],
            'savings_status' : data['savings_status'],
            'employment' : data['employment'],
    
    }])

    prediction = model.predict(input_data)[0]
    probability  = model.predict_proba(input_data)[0][1]

    if probability >= 0.8 : 
        risk_level = "LOW RISK"
    elif probability >= 0.6 :
        risk_level = "MEDIUM RISK"   
    else :
        risk_level = "HIGH RISK"     

    credit_score = int(300 + probability * 550)

    return render_template(
        "index.html",
        prediction = "Good Credit " if prediction == 1 else "Bad Credit",
        probability = round(probability,3),
        risk_level = risk_level,
        credit_score = credit_score
    )        

if __name__ == "__main__" :
    app.run(debug=True)
