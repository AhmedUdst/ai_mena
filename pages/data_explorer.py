import streamlit as st
import pandas as pd
import requests
import io
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="📊 Free Time Data Explorer", layout="wide")

# Theme toggle in sidebar
theme = st.sidebar.radio("🌗 Select Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """
        <style>
        body, .stApp, .css-ffhzg2 {
            background-color: #0e1117;
            color: #e1e1e1;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #f0f0f0;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
        }
        .stSelectbox, .stNumberInput input {
            color: black !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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
        fig = px.bar(df[selected_col].value_counts().reset_index(), 
                     x='index', y=selected_col, 
                     labels={'index': selected_col, selected_col: 'Count'},
                     title=f"Distribution of {selected_col}")
        st.plotly_chart(fig)
    else:
        st.write(df[selected_col].describe())
        fig = px.histogram(df, x=selected_col, marginal="rug", nbins=30,
                           title=f"Distribution of {selected_col}")
        st.plotly_chart(fig)

    # Correlation heatmap
    if st.checkbox("📌 Show correlation heatmap (numeric only)"):
        numeric_df = df.select_dtypes(include='number')
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    # Pairwise relationships
    if st.checkbox("📈 Show Pairplot (slow)"):
        pair_cols = st.multiselect("Select columns for pairplot", df.select_dtypes(include='number').columns.tolist(), default=df.select_dtypes(include='number').columns[:3])
        if len(pair_cols) > 1:
            fig = sns.pairplot(df[pair_cols])
            st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Failed to load dataset from GitHub: {e}")
