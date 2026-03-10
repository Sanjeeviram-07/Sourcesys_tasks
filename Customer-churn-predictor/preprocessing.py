import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = [
'gender','SeniorCitizen','Partner','Dependents','tenure',
'PhoneService','MultipleLines','InternetService','OnlineSecurity',
'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
'StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
'MonthlyCharges','TotalCharges'
]

def preprocess_data(df):

    df = df.drop(columns=["customerID"], errors="ignore")

    # Fix TotalCharges datatype
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.dropna()

    X = df[FEATURE_COLUMNS]
    y = df["Churn"]

    label = LabelEncoder()
    y = label.fit_transform(y)

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(include=["int64","float64"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor, FEATURE_COLUMNS