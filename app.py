import streamlit as st
import pandas as pd
import requests
import io
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(page_title="📊 Free Time Predictor", layout="wide")
st.title("🎓 Predict Free Time Using ML")
st.markdown("""
Welcome to your interactive machine learning dashboard!

- Data is auto-loaded from GitHub
- Select features
- Compare multiple ML models
""")

# Load dataset from GitHub (fixed link)
github_url = "https://raw.githubusercontent.com/yourusername/yourrepo/main/dataset.csv"
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

        # Encode all columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

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

    else:
        st.warning("⚠️ Please select at least one feature column.")

except Exception as e:
    st.error(f"❌ Failed to load dataset from GitHub: {e}")
