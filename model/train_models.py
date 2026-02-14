from model.preprocessing import preprocess_dataframe
from model.logistic_regression import build as lr
from model.decision_tree import build as dt
from model.knn import build as knn
from model.naive_bayes import build as nb
from model.random_forest import build as rf
from model.xgboost_model import build as xgb

def train_all_models(df_train):
  
    # Fit encoders and scaler only on training data
    X_train, X_train_scaled, y_train, encoders, scaler = preprocess_dataframe(
        df_train, fit=True
    )

    models = {
        "Logistic Regression": lr(),
        "Decision Tree": dt(),
        "KNN": knn(),
        "Naive Bayes": nb(),
        "Random Forest": rf(),
        "XGBoost": xgb()
    }

    for name, model in models.items():
        if name in ["Logistic Regression", "KNN", "Naive Bayes"]:
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)

    return models, encoders, scaler

