############################################################
# Author: Laurence Burden
#    For: Purdue University Global
#
# Unit 7 Assignment / Module 5 Part 1 Competency Assessment
# ##########################################################

# Library and data imports
import sys

# Ignore warnings
if not sys.warnoptions:
    import warnings

warnings.simplefilter("ignore")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from statistics import variance

# Allow pandas to output more information
pd.set_option('display.max_columns', None)

# Output header
print('Unit 7 Assignment / Module 5 Part 1 Competency Assessment Output\n')
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S"), '\n')

# Import dataset
df = pd.read_csv('Churn_Modelling.csv')

# Explorative Analysis
print('DATAFRAME HEAD:')
print(df.head(10), '\n')

print('DATAFRAME TAIL:')
print(df.tail(10), '\n')

# Check for missing values
print('ANY DATAFRAME NAs: ', df.isna().values.any(), '\n')

# Check for any null values
print('DATAFRAME NULL TOTAL:')
print(df.isnull().sum(), '\n')

# Print the structure of the data
print('DATAFRAME INFO:')
print(df.info(), '\n')

print('DATAFRAME DESCRIPTION:')
print(df.describe(), '\n')

# Check the variance of each variable
# First: set the attribute values
creditscore = df['CreditScore']
age = df['Age']
tenure = df['Tenure']
balance = df['Balance']
estimatedSalary = df['EstimatedSalary']

# Display variance values
print('Variance of CreditScore is % s ' % (variance(creditscore)), '\n')
print('Variance of Age is % s ' % (variance(age)), '\n')
print('Variance of Tenure is % s ' % (variance(tenure)), '\n')
print('Variance of Balance is % s ' % (variance(balance)), '\n')
print('Variance of EstimatedSalary is % s ' % (variance(estimatedSalary)), '\n')

# Build a plot to visualize customers that churned and that did not churn
# First plot - Between customers that churned and those that did not churn
plt.figure(figsize=(10,5))
sns.countplot(x='Exited', data=df)
plt.show()

# Calculate the percentage of churned customers
total_customers = len(df.index)
customers_churned = df.groupby('Exited').Exited.count()[1]
perc_cust_churned = customers_churned / total_customers

print('Percentage of churned customers: ', perc_cust_churned*100, '%\n')

# Build a histogram of credit scores for all customers
df['CreditScore'].plot.hist(bins=100, figsize=(10,5))

# Identify unique values in the Geography column
df['Geography'].unique()

# Plot the geography for all customers
plt.figure(figsize=(10,5))
sns.countplot(x='Geography', data=df)
plt.show()

# Plot the geography for churned/non-churned customers
plt.figure(figsize=(10,5))
sns.countplot(x='Geography', hue='Exited', data=df)
plt.show()

# Plot the gender by the churn status
plt.figure(figsize=(10,5))
sns.countplot(x='Exited', hue='Gender', data=df)
plt.show()

# Calculate the percentage of customers by gender
churn_by_gender = df.groupby(['Gender'])['Exited'].sum()
print('CHURN BY GENDER')
print(churn_by_gender, '\n')

# Calc churn number by gender (Male)
churned_males = churn_by_gender['Male']
print('Churned males: ', str(churned_males), '\n')

# Calc churn number by gender (Female)
churned_females = churn_by_gender['Female']
print('Churned females: ', str(churned_females), '\n')

# Histogram to compare the age by churn status
plt.figure(figsize=(10, 5))
df['Age'].plot.hist()

# Plot a boxplot to identify the churned/non-churned customers by age
plt.figure(figsize=(10, 5))
sns.boxplot(x='Exited', y='Age', data=df)
plt.ylim(0, 100)
plt.show()

# Plot the tenure for churned/non-churned customers
plt.figure(figsize=(10, 5))
sns.countplot(x='Tenure', hue='Exited', data=df)
plt.show()

# Plot the histogram of balance for all customers
plt.figure(figsize=(10, 5))
df['Balance'].plot.hist()

# Plot a number of products by churned/non-churned status
plt.figure(figsize=(10, 5))
sns.countplot(x='NumOfProducts', hue='Exited', data=df)
plt.show()

# Plot the credit card ownership by churned/non-churned status
plt.figure(figsize=(10, 5))
sns.countplot(x='HasCrCard', hue='Exited', data=df)
plt.show()

# Calculate the credit card ownership by churned/non-churned status
churned_by_cc = df.groupby(['HasCrCard'])['Exited'].sum()
churned_no_cc = churned_by_cc[0]
print('Churned with no credit card: ', str(churned_no_cc), '\n')

# Calculate the credit card ownership by churned/non-churned status
churned_cc = churned_by_cc[1]
print('Churned with credit card: ', str(churned_cc), '\n')

# Plot the active hours for the customers by churned/non-churned status
plt.figure(figsize=(10, 5))
sns.countplot(x='IsActiveMember', hue='Exited', data=df)
plt.show()

# Plot the estimated salary for all customers
df['EstimatedSalary'].plot.hist(bins=10000, figsize=(10, 5))
plt.xlabel('EstimatedSalary')
plt.show()
