# E-Commerce Fraud Detection System

# a. Problem statement  
This project focuses on detecting fraudulent e-commerce transactions using multiple machine learning classification models. The goal is to compare different algorithms and evaluate their effectiveness in identifying fraud.
The project also includes a Streamlit web application that allows users to upload a dataset and evaluate model performance interactively.

## b. Dataset description 
The dataset used contains historical e-commerce transaction records with a binary target variable indicating whether a transaction is fraudulent.
Some columns that are not useful for prediction (such as transaction ID and addresses) were removed during preprocessing.

- Number of instances: 23,634
- Number of features: 15
- Target variable: Is Fraudulent
  - 0 → Valid transaction
  - 1 → Fraud transaction

### Feature Types:
- Numerical features: Transaction amount, account age, transaction frequency, etc.
- Categorical features: Payment method, product category, customer location, device used

Categorical features were label-encoded, and irrelevant identifier columns were removed during preprocessing.

---

## c. Models used: 

The following machine learning models were implemented:

- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Random Forest  
- XGBoost  

All models are implemented as Python source files (`.py`) as required.

### Evaluation Metrics:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Comparison Table with the evaluation metrics calculated for all the models as below:  

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|---------|-----|----------|--------|----------|-----|
| Logistic Regression | 0.953415 | 0.77806 | 0.849711 | 0.120295 | 0.210753 | 0.309469 |
| Decision Tree | 1 | 1 | 1 | 1 | 1 | 1 |
| KNN | 0.95608 | 0.948233 | 0.883333 | 0.173486 | 0.290014 | 0.380403 |
| Naive Bayes | 0.942456 | 0.773368 | 0.396396 | 0.216039 | 0.279661 | 0.265074 |
| Random Forest | 1 | 1 | 1 | 1 | 1 | 1 |
| XGBoost | 0.977109 | 0.99086 | 0.99708 | 0.55892 | 0.716308 | 0.737614 |


---

##  Observations on the performance of each model on the chosen dataset. 

| ML Model | Observation |
|--------|------------|
| Logistic Regression | Performed well as a baseline model but struggled with capturing complex non-linear patterns in fraud data. |
| Decision Tree | Able to model non-linear relationships but prone to overfitting. |
| kNN | Sensitive to feature scaling and computationally expensive for large datasets. |
| Naive Bayes | Fast and efficient but assumes feature independence, which limits performance. |
| Random Forest (Ensemble)  | Achieved strong overall performance by reducing overfitting through ensemble learning. |
| XGBoost (Ensemble)  | Delivered the best performance with high AUC and MCC due to its boosting-based approach and regularization. |
