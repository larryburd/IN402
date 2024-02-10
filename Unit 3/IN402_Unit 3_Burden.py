##############################################################
# Author: Laurence T. Burden
# for Purdue Global University
#
# Unit 3 Assignment / Module 3 Part 1 Competency Assessment
# Predicting Gender-Based Salary Gap
##############################################################

# Imports
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Ignore warnings
if not sys.warnoptions:
    import warnings

warnings.simplefilter("ignore")

# Set Pandas to display more columns when printing
pd.set_option('display.max_columns', 25)

# Output Header
print('Unit 3 Assignment / Module 3 Part 1 Competency Assessment Output\n')
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S"), '\n')

# Import and explore the quality of the dataset
df = pd.read_csv('data.csv')

# Print data types, shape, first few rows, and last few rows
print('### Data Types ###')
print(df.dtypes, '\n')
print('### Data Frame Shape ###')
print(df.shape, '\n')
print('### Data Frame Head ###')
print(df.head(), '\n')
print('### Data Frame Tail ###')
print(df.tail(), '\n')
print('### Data Frame Info ###')
print(df.info(), '\n')

# Check for missing and duplicated values
print('### Data Frame Counts ###')
print(df.count(), '\n')
print('### Duplicates ###')
print(df[df.duplicated(keep=False)], '\n')

# Describe the data
print(df.describe(), '\n')

# Print null hypothesis
print('Null Hypothesis: Gender has no effect on base pay')

femaleInfo = df[df['gender'] == 'Female']
maleInfo = df[df['gender'] == 'Male']
plt.scatter(femaleInfo['age'], femaleInfo['basePay'], color='red', label='Female', alpha=0.8)
plt.scatter(maleInfo['age'], maleInfo['basePay'], color='blue', label='Male', alpha=0.8)
plt.legend()
plt.title('Base Pay Versus Age Split by Men and Women')
plt.show()

# Wrangle the data
# Create dummy variable for gender for use in regression (1 for male and 0 for female)
df = pd.get_dummies(df, columns=['gender', 'edu'])
print(df.head(), '\n')

# Group ages into 5 age groups
labels = ['18-26', '27-36', '37-47', '48-57', '57-65']
df['AgeGroup'] = pd.qcut(df['age'], q=5, labels=labels)
print('### Age Bins ###')
print(df['AgeGroup'].value_counts(), '\n')

# Plot the age groups
plt.hist(df['AgeGroup'], bins=5, align='right')
plt.title('Age Group Counts')
plt.show()

# Natural log of base pay
df['logBasePay'] = np.log(df['basePay'])
print('### Base Pay Log ###')
print(df['logBasePay'].head(), '\n')

# Linear Regression of logBasePay ~ age
sns.regplot(x='age', y='logBasePay', data=df)
plt.title('Linear Regression of Log of Base Pay and Employee Age')
plt.show()

# Linear Regression of logBasePay ~ gender_Female
# reshape data to fit the linear regression method
gender = df['gender_Female'].to_numpy()
logBasePay = df['logBasePay'].to_numpy()
X = gender.reshape(-1, 1)
y = logBasePay.reshape(-1, 1)

# Fit and score the linear regression model
lm = LinearRegression()
lm.fit(X, y)
r2 = lm.score(X, y)

# Print results
print('### Linear Regression of Gender and Log of Base Pay ###')
print('Intercept: ', lm.intercept_)
print('Coefficient: ', lm.coef_)
print('R-Squared: ', r2, '\n')

# Plot the linear regression
sns.regplot(x='gender_Female', y='logBasePay', data=df)
plt.title('Linear Regression of Log of Base Pay and Employee Gender')
plt.legend(['0.0 = Male', '1.0 = Female'], markerscale=0, handlelength=0)
plt.show()

# Run the multiple regression
# Create the model
model = sm.ols(data=df, formula="basePay ~ gender_Female + gender_Male + age + seniority")

# fit the model
result = model.fit()

# View results
print('### OLS of Original Base Pay Data ###')
print(result.summary())

# Recreate the model with the log of base pay
model = sm.ols(data=df, formula="logBasePay ~ gender_Female + gender_Male + age + seniority")

# fit the model
result = model.fit()

# View results
print('### OLS of Natural Log of Base Pay Data ###')
print(result.summary())
