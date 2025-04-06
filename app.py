import streamlit as st
import pandas as pd
import requests
import io
import re
import joblib
import os

st.set_page_config(page_title="📊 Free Time Predictor", layout="wide")
st.title("🎓 Predict Free Time Using ML")
st.markdown("""
Welcome to your interactive machine learning dashboard!

- Load trained model from pickle
- Select features
- Predict free time instantly
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
            st.subheader("📄 Data Preview")
            st.dataframe(df.head())

            st.subheader("🔮 Make a New Prediction")
            user_input = {}
            for col in feature_cols:
                if col in encoders:
                    input_val = st.selectbox(f"{col}", list(encoders[col].classes_))
                    user_input[col] = encoders[col].transform([input_val])[0]
                else:
                    user_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))

            if st.button("Predict Free Time"):
                input_df = pd.DataFrame([user_input])
                prediction = model.predict(input_df)
                predicted_label = label_encoder.inverse_transform(prediction)[0]
                st.success(f"🕒 Predicted Free Time: {predicted_label}")

        else:
            st.warning("⚠️ Dataset does not contain the required columns for prediction.")

except Exception as e:
    st.error(f"❌ Failed to load dataset from GitHub: {e}")
