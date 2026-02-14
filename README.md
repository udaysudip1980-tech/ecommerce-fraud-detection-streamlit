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
| Logistic Regression | 0.953882 | 0.778199 | 0.861111 | 0.127049 | 0.221429 | 0.320507 |
| Decision Tree | 0.897186 | 0.614457 | 0.188144 | 0.29918 | 0.231013 | 0.18452 |
| KNN | 0.950709 | 0.62118  | 0.622222 | 0.114754 | 0.193772 | 0.252832 |
| Naive Bayes | 0.937804 | 0.770417 | 0.328767 | 0.196721 | 0.246154 | 0.223623 |
| Random Forest(Ensemble) | 0.952613 | 0.799907 | 0.692308 | 0.147541 | 0.243243 | 0.305397 |
| XGBoost(Ensemble) | 0.949862 | 0.763527 | 0.54321  | 0.180328 | 0.270769 | 0.293369 |


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
