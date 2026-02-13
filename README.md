# E-Commerce Fraud Detection using Machine Learning

## 1. Problem Statement
Online e-commerce platforms face a significant challenge in detecting fraudulent transactions in real time. The objective of this project is to build and compare multiple machine learning classification models to accurately identify fraudulent e-commerce transactions based on transaction and customer-related features. The project also includes deploying an interactive Streamlit web application for real-time model evaluation.

---

## 2. Dataset Description
The dataset used in this project is a public e-commerce transaction dataset containing information related to customer behavior, transaction details, and payment methods.

- Number of instances: 23,634
- Number of features: 15
- Target variable: **Is Fraudulent**
  - 0 → Legitimate transaction
  - 1 → Fraudulent transaction

### Feature Types:
- Numerical features: Transaction amount, account age, transaction frequency, etc.
- Categorical features: Payment method, product category, customer location, device used

Categorical features were label-encoded, and irrelevant identifier columns were removed during preprocessing.

---

## 3. Models Used and Evaluation Metrics

The following six machine learning classification models were implemented and evaluated on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Evaluation Metrics:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 4. Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|---------|-----|----------|--------|----------|-----|
| Logistic Regression | 0.95367 | 0.777792 | 0.857143 | 0.122951 | 0.215054 | 0.314442 |
| Decision Tree | 0.920669 | 0.630713 | 0.266904 | 0.307377 | 0.285714 | 0.244619 |
| KNN | 0.950709 | 0.624074 | 0.622222 | 0.114754 | 0.193772 | 0.252832 |
| Naive Bayes | 0.938227 | 0.769832 | 0.333333 | 0.196721 | 0.247423 | 0.225695 |
| Random Forest | 0.953036 | 0.808093 | 0.689655 | 0.163934 | 0.264901 | 0.321406 |
| XGBoost | 0.95092 | 0.795012 | 0.565217 | 0.213115 | 0.309524 | 0.327039 |


---

## 5. Observations on Model Performance

| ML Model | Observation |
|--------|------------|
| Logistic Regression | Performed well as a baseline model but struggled with capturing complex non-linear patterns in fraud data. |
| Decision Tree | Able to model non-linear relationships but prone to overfitting. |
| KNN | Sensitive to feature scaling and computationally expensive for large datasets. |
| Naive Bayes | Fast and efficient but assumes feature independence, which limits performance. |
| Random Forest | Achieved strong overall performance by reducing overfitting through ensemble learning. |
| XGBoost | Delivered the best performance with high AUC and MCC due to its boosting-based approach and regularization. |

---

## 6. Streamlit Web Application
A Streamlit web application was developed and deployed to demonstrate the trained models.

### Features:
- CSV file upload (test data only)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix and classification report

The app allows users to interactively analyze fraud detection performance using different machine learning models.

---

## 7. Project Structure
project/
│── app.py
│── requirements.txt
│── README.md
│
└── model/
│── logistic_regression.pkl
│── decision_tree.pkl
│── knn.pkl
│── naive_bayes.pkl
│── random_forest.pkl
│── xgboost.pkl
│── scaler.pkl


---

## 8. Deployment
The application was deployed using Streamlit Community Cloud by connecting the GitHub repository and selecting the app.py file for deployment.

---

## 9. Conclusion
This project demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation, and deployment. Ensemble models such as Random Forest and XGBoost were observed to perform best for fraud detection tasks due to their ability to capture complex patterns in transactional data.
