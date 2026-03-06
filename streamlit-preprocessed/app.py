import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, confusion_matrix, classification_report

st.title("ML Pipeline + GridSearch Dashboard")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # detect column types
    num_cols = X.select_dtypes(include=["int64","float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # numeric pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # categorical pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    model = RandomForestRegressor()

    pipe = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    # Grid parameters
    param_grid = {
        "model__n_estimators": [50,100],
        "model__max_depth": [None,5,10]
    }

    st.subheader("Parameter Grid")
    st.write(list(ParameterGrid(param_grid)))

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    if st.button("Train Model"):

        grid = GridSearchCV(pipe,param_grid,cv=3)

        grid.fit(X_train,y_train)

        best_model = grid.best_estimator_

        st.subheader("Best Parameters")
        st.write(grid.best_params_)

        y_pred = best_model.predict(X_test)

        r2 = r2_score(y_test,y_pred)

        st.subheader("R2 Score")
        st.write(r2)

        y_pred_class = (y_pred > y_pred.mean()).astype(int)
        y_test_class = (y_test > y_test.mean()).astype(int)

        cm = confusion_matrix(y_test_class,y_pred_class)

        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots()
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y_test_class,y_pred_class))