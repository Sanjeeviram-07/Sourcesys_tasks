import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

st.title("ML Model Dashboard")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_option = st.selectbox(
        "Select Model",
        ["Logistic Regression", "KNN"]
    )

    if st.button("Train Model"):

        if model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)

        if model_option == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.subheader("Accuracy Score")
        st.write(acc)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)

        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)