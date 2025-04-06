import streamlit as st
import pandas as pd
import requests
import io
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

st.set_page_config(page_title="📊 Free Time Predictor", layout="wide")
st.title("🎓 Predict Free Time Using ML")
st.markdown("""
Welcome to your interactive machine learning dashboard!

- Data is auto-loaded from GitHub
- Select features
- Compare multiple ML models
- Predict free time using trained model
""")

# Load dataset from GitHub (fixed link)
github_url = "https://raw.githubusercontent.com/AhmedUdst/ai_mena/refs/heads/main/dataset.csv"
try:
    response = requests.get(github_url)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))

    # Clean column names
    df.columns = [re.sub(r"[^\x00-\x7F]+", "", col).strip().lower().replace(" ", "_") for col in df.columns]

    st.success("✅ Dataset loaded from GitHub successfully!")
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # Target and feature selection
    st.sidebar.header("🎯 Select Target and Features")
    target_col = st.sidebar.selectbox("Select target column (e.g., free_time)", df.columns)
    feature_cols = st.sidebar.multiselect("Select feature columns", [col for col in df.columns if col != target_col])

    if feature_cols:
        # Drop NA and reset index
        df = df[[target_col] + feature_cols].dropna().reset_index(drop=True)

        # Save original labels for decoding predictions later
        label_encoder = LabelEncoder()
        df[target_col] = label_encoder.fit_transform(df[target_col].astype(str))
        class_labels = label_encoder.classes_

        # Encode features
        encoders = {}
        for col in feature_cols:
            if df[col].dtype == 'object':
                enc = LabelEncoder()
                df[col] = enc.fit_transform(df[col].astype(str))
                encoders[col] = enc

        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        nb = CategoricalNB()
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        voting_clf = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('nb', nb)], voting='soft')

        models = {
            "Random Forest": rf,
            "Logistic Regression": lr,
            "Naive Bayes": nb,
            "XGBoost": xgb,
            "Voting Classifier": voting_clf
        }

        st.subheader("📊 Model Performance")
        summary = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            summary.append({
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 3),
                "Macro F1": round(f1_score(y_test, y_pred, average='macro'), 3),
                "Weighted F1": round(f1_score(y_test, y_pred, average='weighted'), 3)
            })

        st.dataframe(pd.DataFrame(summary))

        st.subheader("📋 Classification Report (Voting Classifier)")
        voting_clf.fit(X_train, y_train)
        y_pred_voting = voting_clf.predict(X_test)
        report = classification_report(y_test, y_pred_voting, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # Save trained model and encoders
        joblib.dump(voting_clf, "voting_model.pkl")
        joblib.dump(label_encoder, "label_encoder.pkl")
        joblib.dump(encoders, "feature_encoders.pkl")

        st.subheader("🔮 Make a New Prediction")
        user_input = {}
        for col in feature_cols:
            options = df[col].unique().tolist()
            if col in encoders:
                reverse_map = {v: k for k, v in dict(zip(encoders[col].classes_, encoders[col].transform(encoders[col].classes_))).items()}
                input_val = st.selectbox(f"{col}", list(encoders[col].classes_))
                user_input[col] = encoders[col].transform([input_val])[0]
            else:
                user_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))

        if st.button("Predict Free Time"):
            input_df = pd.DataFrame([user_input])
            model = joblib.load("voting_model.pkl")
            label_enc = joblib.load("label_encoder.pkl")
            prediction = model.predict(input_df)
            predicted_label = label_enc.inverse_transform(prediction)[0]
            st.success(f"🕒 Predicted Free Time: {predicted_label}")

    else:
        st.warning("⚠️ Please select at least one feature column.")

except Exception as e:
    st.error(f"❌ Failed to load dataset from GitHub: {e}")
