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
| Logistic Regression | 0.9539 | 0.7782 | 0.8611 | 0.1270 | 0.2214 | 0.3205 |
| Decision Tree | 0.8972 | 0.6145 | 0.1881 | 0.2992 | 0.2310 | 0.1845 |
| KNN | 0.9507 | 0.6212 | 0.6222 | 0.1148 | 0.1938 | 0.2528 |
| Naive Bayes | 0.9378 | 0.7704 | 0.3288 | 0.1967 | 0.2462 | 0.2236 |
| Random Forest(Ensemble) | 0.9526 | 0.7999 | 0.6923 | 0.1475 | 0.2432 | 0.3054 |
| XGBoost(Ensemble) | 0.9499 | 0.7635 | 0.5432 | 0.1803 | 0.2708 | 0.2934 |


---

##  observations on the performance of each model on the chosen dataset. 

| ML Model | Observation |
|--------|------------|
| Logistic Regression | Gave high accuracy but missed many fraud cases, as shown by its very low recall. |
| Decision Tree | Picked up more fraud cases than most models but overall performance was weaker and less reliable. |
| kNN | Showed good accuracy, but struggled badly to identify fraud cases, leading to low recall and F1 score. |
| Naive Bayes | Performed decently overall but made simplifying assumptions that limited its ability to detect fraud accurately. |
| Random Forest(Ensemble)  | Balanced performance well with strong AUC and MCC, showing better handling of complex patterns. |
| XGBoost(Ensemble)  | Delivered the most consistent and effective results, especially in identifying fraud, making it the best overall model. |
