import streamlit as st
import pandas as pd
import requests
import io
import re
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="📊 Free Time Data Explorer", layout="wide")

st.title("📊 Data Exploration Dashboard")

# Load dataset from GitHub
github_url = "https://raw.githubusercontent.com/AhmedUdst/ai_mena/main/clean_dataset.csv"

try:
    response = requests.get(github_url)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))

    # Clean column names
    df.columns = [re.sub(r"[^\x00-\x7F]+", "", col).strip().lower().replace(" ", "_") for col in df.columns]

    # Drop 'id' if exists
    if 'id' in df.columns:
        df.drop(columns='id', inplace=True)

    st.success("✅ Dataset loaded from GitHub successfully!")

    # Dataset preview
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(10))

    # Column selection
    st.subheader("🔍 Explore Columns")
    selected_col = st.selectbox("Select a column to explore", df.columns)

    if df[selected_col].dtype == 'object':
        st.write(df[selected_col].value_counts())
        fig, ax = plt.subplots()
        df[selected_col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f"Distribution of {selected_col}")
        st.pyplot(fig)
    else:
        st.write(df[selected_col].describe())
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {selected_col}")
        st.pyplot(fig)

    # Correlation heatmap
    if st.checkbox("📌 Show correlation heatmap (numeric only)"):
        numeric_df = df.select_dtypes(include='number')
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Failed to load dataset from GitHub: {e}")
