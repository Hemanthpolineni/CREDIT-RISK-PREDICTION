import pandas as pd 
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv(r"C:\Users\polin\Desktop\Python\Projects\Ml_Project\Credit-Risk\german_credit.csv")

df.columns = [
    'checking_status', 'duration', 'credit_history', 'purpose',
    'credit_amount', 'savings_status', 'employment', 'installment_rate',
    'personal_status', 'other_parties', 'residence_since', 'property',
    'age', 'other_payment_plans', 'housing', 'existing_credits', 'job',
    'num_dependents', 'own_telephone', 'foreign_worker', 'target'
]

cols_to_keep = [
    'duration',
    'credit_amount',
    'installment_rate',
    'age',
    'purpose',
    'existing_credits',
    'other_payment_plans',
    'credit_history',
    'checking_status',
    'savings_status',
    'employment',
    'target'
]

df = df[cols_to_keep]

purpose_map = {
    "new car": "vehicle",
    "used car": "vehicle",

    "radio/tv": "consumer_goods",
    "furniture/equipment": "consumer_goods",
    "domestic appliances": "consumer_goods",

    "repairs": "repairs",
    "other": "repairs",

    "education": "education_business",
    "retraining": "education_business",
    "business": "education_business"
}

df["purpose"] = df['purpose'].map(purpose_map)

df["purpose"].fillna("repairs",inplace=True)

credit_map = {
    "no credits/all paid": "good",
    "all paid": "good",
    "existing paid": "good",

    "delayed previously": "bad",
    "critical/other existing credit": "bad"
}

df["credit_history"] = df["credit_history"].map(credit_map)
df["credit_history"].fillna("bad",inplace=True)

X = df.drop("target",axis=1)
y = df["target"]


numcols= [ 
    'duration',
    'credit_amount', 
    'installment_rate',
    'age',
    'existing_credits'
]

cat_cols = [
    "checking_status",
    "credit_history",
    "savings_status",
    "employment",
    "other_payment_plans",
    "purpose"
]    

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)
numerical_transformer = StandardScaler()


Preprocessor = ColumnTransformer (
    transformers=[
        ("numcols",numerical_transformer,numcols),
        ("onehot",categorical_transformer,cat_cols),
        
    ]
)

pipe = Pipeline(steps=[
    ("preprocessor",Preprocessor),
    ("randomforest",RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe.fit(X_train,y_train)

print("Accuracy:", pipe.score(X_test, y_test))

joblib.dump(pipe,"credit_risk.pkl")

print("model created ")