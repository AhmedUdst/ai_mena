import streamlit as st
import pandas as pd
import requests
import io
import re
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

st.set_page_config(page_title="📊 Free Time Predictor", layout="wide")

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

# Page header
st.title("🎓 Predict Free Time Using ML")
st.markdown("""
Welcome to your interactive machine learning dashboard!

- 💾 Load trained model from pickle
- 🎯 Select features
- 🧠 Predict free time instantly
""")

# Load dataset from GitHub
github_url = "https://raw.githubusercontent.com/AhmedUdst/ai_mena/main/dataset.csv"
try:
    response = requests.get(github_url)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))

    # Clean column names
    df.columns = [re.sub(r"[^\x00-\x7F]+", "", col).strip().lower().replace(" ", "_") for col in df.columns]

    # Drop 'id' if it exists
    if 'id' in df.columns:
        df.drop(columns='id', inplace=True)

    # Check if model and encoders exist
    required_files = ["voting_model.pkl", "label_encoder.pkl", "feature_encoders.pkl", "feature_columns.pkl"]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        st.error(f"❌ Missing required model or encoder files: {', '.join(missing_files)}")
    else:
        # Load model and encoders
        model = joblib.load("voting_model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        encoders = joblib.load("feature_encoders.pkl")
        feature_cols = joblib.load("feature_columns.pkl")
        target_col = "free_time"

        # Ensure dataset has necessary columns
        columns_needed = [target_col] + feature_cols
        df = df[[col for col in columns_needed if col in df.columns]].dropna().reset_index(drop=True)

        if all(col in df.columns for col in feature_cols + [target_col]):
            st.success("✅ Dataset loaded from GitHub successfully!")

            # Preview dataset
            with st.expander("📄 Preview Data"):
                st.dataframe(df.head())

            # Layout input columns
            st.subheader("🔮 Make a New Prediction")
            cols = st.columns(3)
            user_input = {}
            for i, col in enumerate(feature_cols):
                with cols[i % 3]:
                    if col in encoders:
                        input_val = st.selectbox(f"{col}", list(encoders[col].classes_))
                        user_input[col] = encoders[col].transform([input_val])[0]
                    else:
                        user_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))

            if st.button("📌 Predict Free Time"):
                input_df = pd.DataFrame([user_input])
                prediction = model.predict(input_df)
                predicted_label = label_encoder.inverse_transform(prediction)[0]
                st.success(f"🕒 Predicted Free Time: {predicted_label}")

                # Evaluation
                y_true = label_encoder.transform(df[target_col])
                X_encoded = df[feature_cols].copy()
                for col in encoders:
                    if col in X_encoded.columns:
                        X_encoded[col] = encoders[col].transform(X_encoded[col])
                y_pred_all = model.predict(X_encoded)

                # Classification report
                st.markdown("### 🧾 Classification Report")
                report = classification_report(y_true, y_pred_all, target_names=label_encoder.classes_, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                # Confusion matrix
                st.markdown("### 🔍 Confusion Matrix")
                fig, ax = plt.subplots(figsize=(6, 4))
                ConfusionMatrixDisplay.from_predictions(y_true, y_pred_all, display_labels=label_encoder.classes_, ax=ax, cmap='Blues')
                st.pyplot(fig)

                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    st.markdown("### 📊 Feature Importance")
                    importances = pd.Series(model.feature_importances_, index=feature_cols)
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    importances.sort_values().plot(kind='barh', ax=ax2)
                    ax2.set_title("Feature Importance")
                    st.pyplot(fig2)

        else:
            st.warning("⚠️ Dataset does not contain the required columns for prediction.")

except Exception as e:
    st.error(f"❌ Failed to load dataset from GitHub: {e}")
