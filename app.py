import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

st.title("📊 ML Model Comparison App")
st.markdown("Upload a CSV file, select your target column, and compare ML classifiers.")

# File uploader
file = st.file_uploader("Upload CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    # Select target column
    target_col = st.selectbox("🎯 Select the target column (what you want to predict)", df.columns)

    # Preprocess
    df.columns = [re.sub(r"[^\x00-\x7F]+", "", col).strip().lower().replace(" ", "_") for col in df.columns]
    categorical_cols = [col for col in df.columns if col != target_col]

    # Drop rows with missing values (or handle differently)
    df = df.dropna()

    # Encode features
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df[target_col] = LabelEncoder().fit_transform(df[target_col].astype(str))

    X = df[categorical_cols]
    y = df[target_col]

    # Train/test split
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

    # Evaluate
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

    st.subheader("📊 Model Comparison Summary")
    st.dataframe(pd.DataFrame(summary))

else:
    st.info("Upload a CSV file to begin.")
