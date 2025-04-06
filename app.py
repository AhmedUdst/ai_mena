import streamlit as st
import pandas as pd
import requests
import io
import re
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

st.set_page_config(page_title="📊 Free Time Predictor", layout="wide")

# Theme toggle
theme = st.sidebar.radio("🌗 Select Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
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
    """, unsafe_allow_html=True)

# Title
st.title("🎓 Predict Free Time Using ML")
st.markdown("""
Welcome to your interactive machine learning dashboard!

- 💾 Load trained model from pickle
- 🎯 Select features
- 🧠 Predict free time instantly
""")

# GitHub dataset URL
github_url = "https://raw.githubusercontent.com/AhmedUdst/ai_mena/main/dataset.csv"

try:
    response = requests.get(github_url)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))

    # ✅ Clean column names
    df.columns = [re.sub(r"[^\x00-\x7F]+", "", col).strip().lower().replace(" ", "_") for col in df.columns]

    # ✅ Drop ID if present
    if 'id' in df.columns:
        df.drop(columns='id', inplace=True)

    # ✅ Load model and encoders
    required_files = ["nb_model.pkl", "label_encoder.pkl", "feature_encoders.pkl", "feature_columns.pkl"]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        st.error(f"❌ Missing required files: {', '.join(missing_files)}")
    else:
        model = joblib.load("nb_model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        encoders = joblib.load("feature_encoders.pkl")
        feature_cols = joblib.load("feature_columns.pkl")
        target_col = "free_time"

        # ✅ Subset columns
        df = df[[col for col in [target_col] + feature_cols if col in df.columns]].dropna().reset_index(drop=True)

        # ✅ Check column alignment
        if all(col in df.columns for col in feature_cols + [target_col]):
            st.success("✅ Dataset loaded successfully!")

            with st.expander("📄 Preview Data"):
                st.dataframe(df.head())

            # 🔮 Input UI
            st.subheader("🔮 Make a New Prediction")
            cols = st.columns(3)
            user_input = {}
            for i, col in enumerate(feature_cols):
                with cols[i % 3]:
                    input_val = st.selectbox(f"{col}", list(encoders[col].classes_))
                    user_input[col] = encoders[col].transform([input_val])[0]

            # 🔍 Prediction
            if st.button("📌 Predict Free Time"):
                input_df = pd.DataFrame([user_input])
                prediction = model.predict(input_df)
                predicted_label = label_encoder.inverse_transform(prediction)[0]
                st.success(f"🕒 Predicted Free Time: {predicted_label}")

                # 🔎 Evaluation
                y_all = label_encoder.transform(df[target_col])
                X_all = df[feature_cols].copy()
                for col in encoders:
                    if col in X_all.columns:
                        X_all[col] = encoders[col].transform(X_all[col])
                X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
                y_pred = model.predict(X_test)

                st.markdown("### 🧾 Classification Report")
                report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                st.markdown("### 🔍 Confusion Matrix")
                fig, ax = plt.subplots(figsize=(6, 4))
                ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=label_encoder.classes_, ax=ax, cmap='Blues')
                st.pyplot(fig)

        else:
            st.warning("⚠️ Dataset is missing required columns!")

except Exception as e:
    st.error(f"❌ Failed to load dataset from GitHub: {e}")
