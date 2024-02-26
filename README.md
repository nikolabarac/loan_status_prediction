# Loan Status Prediction Project

This project aims to predict the loan status of individuals based on various features. The dataset used for this project was obtained from Kaggle.com. Emphasizing a practical approach, this project prioritizes technical implementation over theoretical intricacies. As a beginner, every suggestion and feedback are highly appreciated.

## Data Preprocessing

1. **Data Splitting:**
   - The dataset was split into training and test sets, using a stratified approach based on 'Credit Score'. This ensures that the distribution of 'Credit Score' remains consistent in both sets.

2. **Feature Engineering:**
   - Categorical and numerical features were identified and treated separately.
   - Categorical features were encoded using techniques such as one-hot encoding or label encoding.
   - Numerical features were scaled to bring them to a similar scale.

## Classification Models

Various classification models were employed to predict loan status. The models include:

- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors Classifier
- Gaussian Naive Bayes Classifier
- Support Vector Classifier
- Logistic Regression
- XGBoost Classifier

## Model Evaluation

Model performance was evaluated using the following metrics:

- **Accuracy:** The proportion of correctly classified instances.
- **Precision/Recall:** Precision measures the accuracy of the positive predictions, while recall measures the ability to capture all positive instances.
- **F1 Score:** The harmonic mean of precision and recall.
- **ROC Curve:** Receiver Operating Characteristic curve and Area Under the Curve (AUC) were used to assess the model's ability to distinguish between classes.

## Acknowledgments
https://www.kaggle.com/
