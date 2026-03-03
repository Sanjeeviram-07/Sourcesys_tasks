import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

# --- 1. Data Loading & Preprocessing ---
@st.cache_data
# --- 1. Data Loading & Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    # Load inbuilt dataset
    df = sns.load_dataset('titanic')
    
    # FIX: Added 'alone' to the dropped columns list
    df = df.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'], axis=1)
    
    # Handle missing values
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder # Ensure this is imported at the top
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    df['sex'] = le_sex.fit_transform(df['sex'])
    df['embarked'] = df['embarked'].astype(str) # Catch any lingering NaNs just in case
    df['embarked'] = le_embarked.fit_transform(df['embarked'])
    
    return df, le_sex, le_embarked

# --- 2. Model Training ---
@st.cache_resource
def train_model(df):
    X = df.drop('survived', axis=1)
    y = df['survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions for evaluation
    y_pred = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, acc, report, cm, X.columns

# --- Main App Execution ---
st.title("🚢 Titanic Survival Predictor & Evaluator")
st.write("This app trains a Random Forest model on the inbuilt Titanic dataset, evaluates its performance, and lets you test custom passenger profiles.")

# Load Data and Train Model
df, le_sex, le_embarked = load_and_preprocess_data()
model, accuracy, report, cm, feature_names = train_model(df)

# --- App Layout: Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Model Evaluation", "🔮 Make a Prediction", "📂 Dataset View"])

with tab1:
    st.header("Performance Metrics")
    st.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())
        
    with col2:
        st.subheader("Confusion Matrix")
        st.write(cm)
        st.caption("Top-Left: True Negatives | Top-Right: False Positives")
        st.caption("Bottom-Left: False Negatives | Bottom-Right: True Positives")

with tab2:
    st.header("Predict Passenger Survival")
    st.write("Adjust the parameters below to see if your hypothetical passenger would survive.")
    
    # User inputs for prediction
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3], index=2)
        sex_input = st.radio("Sex", ['male', 'female'])
        age = st.number_input("Enter your age", min_value=1, max_value=120, value=25)
        
    with col_p2:
        sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
        parch = st.number_input("Parents/Children Aboard (Parch)", 0, 6, 0)
        fare = st.number_input("Fare Paid", min_value=0.0, max_value=500.0, value=32.0)
        embarked_input = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

    # Format input for prediction
    encoded_sex = le_sex.transform([sex_input])[0]
    encoded_embarked = le_embarked.transform([embarked_input])[0]
    
    input_data = pd.DataFrame([[pclass, encoded_sex, age, sibsp, parch, fare, encoded_embarked]], 
                              columns=feature_names)
    
    # Prediction Button
    if st.button("Predict Survival", type="primary"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        if prediction == 1:
            st.success(f"**Survived!** (Probability: {probability:.2%})")
            st.balloons()
        else:
            st.error(f"**Did not survive.** (Survival Probability: {probability:.2%})")

with tab3:
    st.header("Preprocessed Dataset")
    st.write("A quick look at the data used to train the model after cleaning and encoding.")
    st.dataframe(df.head(100))