import streamlit as st
import pandas as pd
import numpy as np

from joblib import load
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": load("model/logistic_regression.pkl"),
        "Decision Tree": load("model/decision_tree.pkl"),
        "KNN": load("model/knn.pkl"),
        "Naive Bayes": load("model/naive_bayes.pkl"),
        "Random Forest": load("model/random_forest.pkl"),
        "XGBoost": load("model/xgboost.pkl")
    }
    scaler = load("model/scaler.pkl")
    return models, scaler

st.set_page_config(page_title="Fraud Detection App", layout="wide")

st.title("💳 E-Commerce Fraud Detection")
st.write("""
This Streamlit app allows you to:
- Upload transaction test data (CSV)
- Select a machine learning model
- View evaluation metrics
- Analyze fraud prediction performance
""")

uploaded_file = st.file_uploader(
    "Upload test dataset (CSV only)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(df.head())

    drop_cols = [
        "Transaction ID",
        "Customer ID",
        "IP Address",
        "Shipping Address",
        "Billing Address",
        "Transaction Date"
    ]

    df = df.drop(columns=drop_cols, errors="ignore")

    X = df.drop("Is Fraudulent", axis=1)
    y = df["Is Fraudulent"]

    models, scaler = load_models()

    selected_model_name = st.selectbox(
        "Select Machine Learning Model",
        list(models.keys())
    )

    model = models[selected_model_name]

    if selected_model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X


    y_pred = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled)[:, 1]
    else:
        y_prob = y_pred

    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy_score(y, y_pred), 4))
    col1.metric("AUC", round(roc_auc_score(y, y_prob), 4))

    col2.metric("Precision", round(precision_score(y, y_pred), 4))
    col2.metric("Recall", round(recall_score(y, y_pred), 4))

    col3.metric("F1 Score", round(f1_score(y, y_pred), 4))
    col3.metric("MCC", round(matthews_corrcoef(y, y_pred), 4))

    st.subheader("🧮 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)


    st.subheader("📄 Classification Report")
    st.text(classification_report(y, y_pred))

else:
    st.warning("Please upload a CSV file to proceed.")

