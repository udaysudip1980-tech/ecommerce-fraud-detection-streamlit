
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
# Import your own PY-based modules
# --------------------------------------------------
from model.train_models import train_all_models
from model.preprocessing import preprocess_dataframe

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Fraud Detection",
    layout="wide"
)

# --------------------------------------------------
# App Title & Description
# --------------------------------------------------
st.title("💳 E-Commerce Fraud Detection System")

st.write("""
This Streamlit application demonstrates multiple machine learning
classification models for detecting fraudulent e-commerce transactions.

**Features:**
- Upload test dataset (CSV)
- Select ML model
- View evaluation metrics
- Confusion matrix & classification report
""")

# --------------------------------------------------
# Load & Train Models ONCE (Cached in Memory)
# --------------------------------------------------
@st.cache_resource
def load_models_and_preprocessors():
    """
    Trains all models once at application startup.
    Uses only .py model implementations (no .pkl).
    """
    df_train = pd.read_csv("Fraudulent_E-Commerce_Transaction_Data_2.csv")
    models, encoders, scaler = train_all_models(df_train)
    return models, encoders, scaler


models, encoders, scaler = load_models_and_preprocessors()

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload test dataset (CSV only)",
    type=["csv"]
)

# --------------------------------------------------
# Main App Logic
# --------------------------------------------------
if uploaded_file is not None:

    df_test = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(df_test.head())

    # --------------------------------------------------
    # Preprocess uploaded dataset (NO fitting)
    # --------------------------------------------------
    try:
        X, X_scaled, y, _, _ = preprocess_dataframe(
            df_test,
            fit=False,
            encoders=encoders,
            scaler=scaler
        )
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    # --------------------------------------------------
    # Model Selection
    # --------------------------------------------------
    st.subheader("⚙️ Model Selection")

    selected_model_name = st.selectbox(
        "Select Machine Learning Model",
        list(models.keys())
    )

    model = models[selected_model_name]

    # --------------------------------------------------
    # Select Correct Feature Set
    # --------------------------------------------------
    if selected_model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X_input = X_scaled
    else:
        X_input = X

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------
    y_pred = model.predict(X_input)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_input)[:, 1]
    else:
        y_prob = y_pred

    # --------------------------------------------------
    # Display Metrics
    # --------------------------------------------------
    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy_score(y, y_pred), 4))
    col1.metric("AUC", round(roc_auc_score(y, y_prob), 4))

    col2.metric("Precision", round(precision_score(y, y_pred), 4))
    col2.metric("Recall", round(recall_score(y, y_pred), 4))

    col3.metric("F1 Score", round(f1_score(y, y_pred), 4))
    col3.metric("MCC", round(matthews_corrcoef(y, y_pred), 4))

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    st.subheader("🧮 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------
    st.subheader("📄 Classification Report")
    st.text(classification_report(y, y_pred))

else:
    st.warning("Please upload a CSV file to begin.")
