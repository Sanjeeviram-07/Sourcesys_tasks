import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

st.title("ML Preprocessing + Model Dashboard")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # Encode target if categorical
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Identify column types
    num_cols = X.select_dtypes(include=["int64","float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # Numeric pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    model_option = st.selectbox(
        "Select Model",
        ["Logistic Regression","KNN"]
    )

    if st.button("Train Model"):

        if model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = KNeighborsClassifier()

        pipe = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.subheader("Accuracy Score")
        st.write(acc)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)

        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        st.pyplot(fig)