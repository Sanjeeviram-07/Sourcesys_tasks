import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from preprocessing import preprocess_data, FEATURE_COLUMNS
from model import build_model

st.title("Customer Churn Prediction Dashboard")

uploaded_file = st.file_uploader("Upload Telco Churn Dataset", type=["csv"])


class StreamlitCallback(Callback):

    def __init__(self, progress_bar, epoch_text, total_epochs):
        self.progress_bar = progress_bar
        self.epoch_text = epoch_text
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):

        progress = (epoch+1)/self.total_epochs
        self.progress_bar.progress(progress)

        self.epoch_text.text(
            f"Epoch {epoch+1}/{self.total_epochs} | "
            f"Loss: {logs['loss']:.4f} | "
            f"Accuracy: {logs['accuracy']:.4f}"
        )


if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    X, y, preprocessor, FEATURE_COLUMNS = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.subheader("Training Configuration")

    epochs = st.slider("Epochs",10,100,30)
    batch_size = st.selectbox("Batch Size",[16,32,64])

    if st.button("Train Model"):

        progress_bar = st.progress(0)
        epoch_text = st.empty()

        model = build_model(X_train.shape[1])

        callback = StreamlitCallback(progress_bar, epoch_text, epochs)

        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[callback],
            verbose=0
        )

        st.success("Training Completed")

        preds = (model.predict(X_test) > 0.5).astype(int)

        acc = accuracy_score(y_test, preds)

        st.subheader("Model Accuracy")
        st.write(acc)

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, preds)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        st.pyplot(fig)

        st.subheader("Classification Report")
        report = classification_report(y_test, preds)
        st.text(report)

        st.subheader("Training Loss Graph")

        fig2, ax2 = plt.subplots()
        ax2.plot(history.history["loss"], label="Train Loss")
        ax2.plot(history.history["val_loss"], label="Validation Loss")
        ax2.legend()

        st.pyplot(fig2)

        st.subheader("Training Accuracy Graph")

        fig3, ax3 = plt.subplots()
        ax3.plot(history.history["accuracy"], label="Train Accuracy")
        ax3.plot(history.history["val_accuracy"], label="Validation Accuracy")
        ax3.legend()

        st.pyplot(fig3)

        os.makedirs("saved_model", exist_ok=True)
        model.save("saved_model/churn_model.h5")

        st.download_button(
            label="Download Trained Model",
            data=open("saved_model/churn_model.h5","rb").read(),
            file_name="churn_model.h5"
        )

        st.session_state["model"] = model
        st.session_state["preprocessor"] = preprocessor


st.header("Predict Customer Churn")

if "model" in st.session_state:

    tenure = st.number_input("Tenure",1,72)
    monthly = st.number_input("Monthly Charges",10,150)

    gender = st.selectbox("Gender",["Male","Female"])

    contract = st.selectbox(
        "Contract",
        ["Month-to-month","One year","Two year"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL","Fiber optic","No"]
    )

    payment = st.selectbox(
        "Payment Method",
        ["Electronic check","Credit card","Bank transfer"]
    )

    if st.button("Predict"):

        input_df = pd.DataFrame({

            "gender":[gender],
            "SeniorCitizen":[0],
            "Partner":["Yes"],
            "Dependents":["No"],
            "tenure":[int(tenure)],
            "PhoneService":["Yes"],
            "MultipleLines":["No"],
            "InternetService":[internet],
            "OnlineSecurity":["No"],
            "OnlineBackup":["Yes"],
            "DeviceProtection":["No"],
            "TechSupport":["No"],
            "StreamingTV":["No"],
            "StreamingMovies":["No"],
            "Contract":[contract],
            "PaperlessBilling":["Yes"],
            "PaymentMethod":[payment],
            "MonthlyCharges":[float(monthly)],
            "TotalCharges":[float(tenure*monthly)]

        })

        input_df = input_df[FEATURE_COLUMNS]

        processed = st.session_state["preprocessor"].transform(input_df)

        prediction = st.session_state["model"].predict(processed)

        if prediction[0] > 0.5:
            st.error("Customer will Churn")
        else:
            st.success("Customer will Stay")