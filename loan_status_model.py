# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:42:30 2024

@author: Nikola
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

X_train = pd.read_csv('D:/Python_e/loan_status_prediction/loan_data_transformed.csv')
y_train = pd.read_csv('D:/Python_e/loan_status_prediction/loan_data_train_labels.csv')

y_train1 = y_train.values.ravel()

X_train.drop('Loan_ID', axis=1, inplace=True)

# Decision Trees

dt_classifier = DecisionTreeClassifier()

print("DecisionTreeClassifier: " + str(np.mean(cross_val_score(dt_classifier, X_train, y_train1, cv=3, scoring="accuracy"))))

# Random Forests

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier()

print("RandomForestClassifier: " + str(np.mean(cross_val_score(rf_classifier, X_train, y_train1, cv=3, scoring="accuracy"))))

# k-Nearest Neighbors (KNN)

from sklearn.neighbors import KNeighborsClassifier


knn_classifier = KNeighborsClassifier()

print("KNeighborsClassifier: " + str(np.mean(cross_val_score(knn_classifier, X_train, y_train1, cv=3, scoring="accuracy"))))

# Naive Bayes

from sklearn.naive_bayes import GaussianNB


nb_classifier = GaussianNB()

print("GaussianNB: " + str(np.mean(cross_val_score(nb_classifier, X_train, y_train1, cv=3, scoring="accuracy"))))

# Support Vector Machines (SVM)

from sklearn.svm import SVC

svm_classifier = SVC()

print("SVC: " + str(np.mean(cross_val_score(svm_classifier, X_train, y_train1, cv=3, scoring="accuracy"))))

# Logistic Regression

from sklearn.linear_model import LogisticRegression

lr_classifier = LogisticRegression()

print("LogisticRegression: " + str(np.mean(cross_val_score(lr_classifier, X_train, y_train1, cv=3, scoring="accuracy"))))

# Gradient Boosting (XGBoost example)

import xgboost as xgb

xgb_classifier = xgb.XGBClassifier()

print("XGBClassifier: " + str(np.mean(cross_val_score(lr_classifier, X_train, y_train1, cv=3, scoring="accuracy"))))

# Confusion matrices

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import seaborn as sns


classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    SVC(),
    LogisticRegression(),
    xgb.XGBClassifier()
]

def evaluate_classifier(classifier, X, y):

    y_train_pred = cross_val_predict(classifier, X, y, cv=3)
    
   
    cm = confusion_matrix(y, y_train_pred)
    
  
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Confusion Matrix - {type(classifier).__name__}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


for classifier in classifiers:
    evaluate_classifier(classifier, X_train, y_train1)
    

# Precision, Recall, F1

from sklearn.metrics import precision_score, recall_score, f1_score


classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    SVC(),
    LogisticRegression(),
    xgb.XGBClassifier()
]

def evaluate_classifier(classifier, X, y):
   
    y_train_pred = cross_val_predict(classifier, X, y, cv=3)
    
    
    precision = precision_score(y, y_train_pred)
    recall = recall_score(y, y_train_pred)
    f1 = f1_score(y, y_train_pred)
    
    print(f"{type(classifier).__name__} Precision: {precision:.4f}")
    print(f"{type(classifier).__name__} Recall: {recall:.4f}")
    print(f"{type(classifier).__name__} F1-Score: {f1:.4f}")

for classifier in classifiers:
    evaluate_classifier(classifier, X_train, y_train1)


# ROC curve

from sklearn.metrics import roc_curve, roc_auc_score

classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    SVC(probability=True),  
    LogisticRegression(),
    xgb.XGBClassifier()
]

def evaluate_classifier(classifier, X, y):
    # Perform cross-validation
    y_prob = cross_val_predict(classifier, X, y, cv=3, method='predict_proba')[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{type(classifier).__name__}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve - {type(classifier).__name__}')
    plt.legend()
    plt.show()

    # Calculate AUC
    auc = roc_auc_score(y, y_prob)
    
    # Print AUC
    print(f"{type(classifier).__name__} AUC: {auc:.4f}")

# Evaluate each classifier
for classifier in classifiers:
    evaluate_classifier(classifier, X_train, y_train1)
    
    
# Checking performance on test set

strat_test_set = pd.read_csv('D:/Python_e/loan_status_prediction/strat_test_set.csv')

strat_test_set.drop('Loan_ID', axis=1, inplace=True)
strat_test_set = pd.get_dummies(strat_test_set, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])

from sklearn.preprocessing import StandardScaler

numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
numerical_data = strat_test_set[numerical_columns]
scaler = StandardScaler()

# Fit and transform the numerical data
numerical_data_standardized = scaler.fit_transform(numerical_data)

# Create a new DataFrame with standardized numerical columns
loan_data_standardized = pd.DataFrame(numerical_data_standardized, columns=numerical_columns)

strat_test_set = strat_test_set.drop(columns=numerical_columns)

loan_data_transformed = strat_test_set.join(loan_data_standardized)

loan_data_transformed.info()

y_test = loan_data_transformed['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)

X_test = loan_data_transformed


# Assuming X_test and y_test are your test set features and labels

# Create a DataFrame to store the results
results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])

# Evaluate each classifier on the test set
for classifier in classifiers:
    # Fit the model on the training set (if not already trained)
    classifier.fit(X_train, y_train1)
    
    # Predict on the test set
    y_pred = classifier.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # For models that provide probability estimates, calculate ROC AUC
    if hasattr(classifier, "predict_proba"):
        y_prob = classifier.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = None
    
    # Append results to the DataFrame
    results_df = results_df.append({
        'Model': type(classifier).__name__,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }, ignore_index=True)

# Display the results
print(results_df)

if X_test is not None:
    print(X_test.isnull().sum())
else:
    print("X_test is None. Please check your data splitting process.")