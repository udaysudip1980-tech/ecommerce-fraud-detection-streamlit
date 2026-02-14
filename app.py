
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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

# --------------------------------------------------
# Import PY-based model pipeline
# --------------------------------------------------
from model.train_models import train_all_models
from model.preprocessing import preprocess_dataframe

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Fraud Detection",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("💳 E-Commerce Fraud Detection System")

st.write("""
This application trains machine learning models on historical e-commerce data
and evaluates fraud detection performance on an uploaded dataset.
""")

# Train models once when the app starts
# This avoids retraining every time Streamlit refreshes

@st.cache_resource
def load_trained_models():
    df = pd.read_csv("Fraudulent_E-Commerce_Transaction_Data_2.csv")

    df_train, _ = train_test_split(
        df,
        test_size=0.2,
        stratify=df["Is Fraudulent"],
        random_state=42
    )

    models, encoders, scaler = train_all_models(df_train)
    return models, encoders, scaler


models, encoders, scaler = load_trained_models()

# Upload a CSV file to test the trained models
# Uploaded data is treated as unseen test data

uploaded_file = st.file_uploader(
    "Upload test dataset (CSV)",
    type=["csv"]
)

# --------------------------------------------------
# App Logic
# --------------------------------------------------
if uploaded_file is not None:

    df_test = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(df_test.head())

# Preprocess uploaded test data
# Important: encoders and scaler should NOT be fitted again

    try:
        X_test, X_test_scaled, y_test, _, _ = preprocess_dataframe(
            df_test,
            fit=False,
            encoders=encoders,
            scaler=scaler
        )
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

# Let the user choose which model to evaluate

    st.subheader("⚙️ Select Model")

    selected_model_name = st.selectbox(
        "Choose Machine Learning Model",
        list(models.keys())
    )

    model = models[selected_model_name]

# Some models require scaled features, others do not

    if selected_model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X_input = X_test_scaled
    else:
        X_input = X_test

# Generate predictions on the test data

    y_pred = model.predict(X_input)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_input)[:, 1]
    else:
        y_prob = y_pred

# Display common classification metrics

    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))
    col1.metric("AUC", round(roc_auc_score(y_test, y_prob), 4))

    col2.metric("Precision", round(precision_score(y_test, y_pred), 4))
    col2.metric("Recall", round(recall_score(y_test, y_pred), 4))

    col3.metric("F1 Score", round(f1_score(y_test, y_pred), 4))
    col3.metric("MCC", round(matthews_corrcoef(y_test, y_pred), 4))

# Confusion matrix to visualize correct vs incorrect predictions

    st.subheader("🧮 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

# Detailed classification report for the selected model

    st.subheader("📄 Classification Report")
    st.text(classification_report(y_test, y_pred))

else:
    st.info("Upload a CSV file to evaluate fraud detection models.")

