import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
sns.set_style("whitegrid")

st.title("🎓 Student Performance Analyzer")

uploaded_file = st.file_uploader("Upload Student CSV File", type=["csv"])

if uploaded_file is not None:

    # Read CSV safely
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    df.columns = df.columns.str.strip()

    st.subheader("📄 Raw Data")
    st.write(df.head())

    # -----------------------
    # FEATURE ENGINEERING
    # -----------------------
    subject_cols = df.select_dtypes(include=np.number).columns

    df["Total"] = df[subject_cols].sum(axis=1)
    df["Average"] = df[subject_cols].mean(axis=1)

    df["Result"] = np.where(df["Average"] >= 40, "Pass", "Fail")

    # -----------------------
    # KPI SECTION
    # -----------------------
    st.subheader("📌 Key Insights")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Average Class Score", f"{df['Average'].mean():.2f}")
    col3.metric("Pass Percentage", f"{(df['Result'].value_counts(normalize=True).get('Pass',0)*100):.1f}%")

    # -----------------------
    # PASS / FAIL DISTRIBUTION

    st.subheader("📊 Visual Insights")

    col1, col2 = st.columns(2)

    # PASS / FAIL DISTRIBUTION
    with col1:
        st.markdown("**Pass / Fail Distribution**")
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        sns.countplot(x="Result", data=df, ax=ax1)
        plt.tight_layout()
        st.pyplot(fig1)

    # SUBJECT AVERAGE
    with col2:
        st.markdown("**Subject-wise Average**")
        subject_avg = df[subject_cols].mean()
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sns.barplot(x=subject_avg.index, y=subject_avg.values, ax=ax2)
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)


    # SECOND ROW
    col3, col4 = st.columns(2)

    # CORRELATION HEATMAP
    with col3:
        st.markdown("**Correlation Heatmap**")
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        sns.heatmap(df[subject_cols].corr(), annot=True, cmap="coolwarm", ax=ax3)
        plt.tight_layout()
        st.pyplot(fig3)

    # AVERAGE SCORE DISTRIBUTION
    with col4:
        st.markdown("**Average Score Distribution**")
        fig4, ax4 = plt.subplots(figsize=(4, 3))
        sns.histplot(df["Average"], bins=10, kde=True, ax=ax4)
        plt.tight_layout()
        st.pyplot(fig4)