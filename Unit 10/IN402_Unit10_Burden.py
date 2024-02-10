#######################################################
# Author: Laurence Burden
# For: Purdue University Global
# IN402 - Modeling and Predictive Analysis
#
# Unit 10 Assignment / Module 6 Part 3 Competency Assessment
#
# Classification Model Selection
###################################################################

# Import packages
import sys
import pandas as pd
from datetime import datetime
from statistics import variance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Ignoring warnings
if not sys.warnoptions:
    import warnings

warnings.simplefilter("ignore")

# Output Header
print('Unit 10 Assignment / Module 6 Part 3 Competency Assessment Output\n')

print(datetime.now().strftime("%m/%d/%Y %H:%M:%S"), '\n')

# Import the dataset
# Importing the dataset to a pandas DataFrame
df = pd.read_csv('Churn_Modelling.csv')
print('Shape of Dataset: ')
print(df.shape)
print()

# Wrangle the data
# Drop columns with no analytical value
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Convert the categorical columns into dummy columns
# and drop the original categorical columns
geography = pd.get_dummies(df.Geography).iloc[:, 1:]
gender = pd.get_dummies(df.Gender).iloc[:, 1:]

# Drop columns with non-numeric data
df = df.drop(['Geography', 'Gender'], axis=1)

# Join the dummy columns into the main dataset
# Add columns with converted dummy values
df = pd.concat([df, geography, gender], axis=1)

# Split the dataset into target and feature subsets.
X = df.drop(['Exited'], axis=1)
y = df.loc[:, 'Exited']

# Select features
# Check the variance in the numeric variables
creditScore = df['CreditScore']
age = df['Age']
tenure = df['Tenure']
balance = df['Balance']
estimatedSalary = df['EstimatedSalary']

# Display the parameter variances
print("Variance of CreditScore is %s " % (variance(creditScore)))
print("Variance of Age is %s " % (variance(age)))
print("Variance of Tenure is %s " % (variance(tenure)))
print("Variance of Balance is %s " % (variance(balance)))
print("Variance of EstimatedSalary is %s " % (variance(estimatedSalary)))
print()

# Split the dataset into training and testing subsets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Display size of training set
print('X_train Length: %s' % (len(X_train)))

# Display size of test set
print('X_test Length: %s' % (len(X_test)))
print()

# Conduct feature scaling (required by SVM)
# feature scaling is required by SVC
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Model using Logistic Regression
# Build the model
lr_model = LogisticRegression()

# Fit the model
result = lr_model.fit(X_train, y_train)

# Predict using the model
prediction_test = lr_model.predict(X_test)

# Evaluate the model
# Print the prediction accuracy
print('LR Prediction Accuracy: ')
print(metrics.accuracy_score(y_test, prediction_test))
print()

# Display the confusion matrix
print('LR Confusion Matrix: ')
print(confusion_matrix(y_test, prediction_test))
print()

# Display the classification report
print('LR Classification Report: ')
print(classification_report(y_test, prediction_test))
print()

# To get the weights of all the variables
weights = pd.Series(lr_model.coef_[0], index=X.columns.values)
weights.sort_values(ascending=False)

# Model using SVM
# Build the model
svm_model = SVC(kernel="linear")

# Fit the model
# Train the model
svm_model.fit(X_train, y_train)

# Predict using the model
print('Predict with SVM Model:')
svm_prediction = svm_model.predict(X_test)
print()

# Evaluate the model
print("accuracy: ", metrics.accuracy_score(y_test, y_pred=svm_prediction))

# Precision score
print("precision: ", metrics.precision_score(y_test, y_pred=svm_prediction))

# Recall score
print("recall", metrics.recall_score(y_test, y_pred=svm_prediction))
print()

# Display classification report
print('SVM Classification Report: ')
print(metrics.classification_report(y_test, y_pred=svm_prediction))
print()

# Model using RandomForestClassifier
# Build the model
rf_model = RandomForestClassifier(n_estimators=200, random_state=0)

# Fit the model
rf_model.fit(X_train, y_train)

# Predict using the model
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
print('RF Evaluate Model: ')
print(classification_report(y_test, rf_predictions))
print()

# Display the accuracy score
print('Accuracy Score: ')
print(accuracy_score(y_test, rf_predictions))
print()

# Display the accuracy score
plt.figure(figsize=(12, 6))
feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(title="Accuracy Score", kind='barh')
plt.show()
