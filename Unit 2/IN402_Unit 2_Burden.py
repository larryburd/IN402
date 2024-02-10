#
# Laurence Burden for Purdue University Global
#
# Unit 2 Assignment / Module 2 Competency Assessment
#

# Imports
import sys

# Ignore warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Output header
print('Unit 2 Assignment / Module 2 Competency Assessment Output\n')

from datetime import datetime

print(datetime.now().strftime("%m/%d/%Y %H:%M:%S"), '\n')

# Load Data
xls = pd.ExcelFile("/home/codio/workspace/data/IN402/NATURALGAS.xls")

# In ts, a Timeseries is the type of index.
# To convert df to ts, make Date column an index
df = xls.parse(0, skiprows=10, index_col=0, na_values=['NA'])

# Plot the Graph
plt.figure(figsize=(10, 6))
plt.grid(True)
plt.xlabel('Month')
plt.ylabel('Natural Gas Consumption, Billion Cubic Feet')
plt.plot(df['NATURALGAS'])
plt.title('Natural Gas Consumption, Monthly')
plt.show()

# Check if the series are stationary
# Determining Rolling Statistics
rolmean = df.rolling(window=12).mean()
rolstd = df.rolling(window=12).std()

print("ROLLING 12-MONTH MEAN")
print(rolmean.head(20))

# Plot rolling statistics
plt.figure(figsize=(10,6))
orig = plt.plot(df, color='blue', label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show()

# Another option - Dickey-Fuller test
# The Dickey-Fuller test can be used to determine
# the presence of unit root in the series (helps us
# understand if the series is stationary)

# Performing Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller

dftest = adfuller(df['NATURALGAS'], autolag='AIC') # Akake Information Criterion
dfoutput = pd.Series(dftest[0:4], index=['Test Statistics', 'p-value', '#Lags Used', 'Number of Observations Used'])

for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value

print("Results of Dickey-Fuller Test: ")
print(dfoutput)
print()

# Transform data using the log of the time series and
# re-calculate the Dickey-Fuller test again on the transformed data

# Estimating trend
plt.figure(figsize=(10, 6))
df_logScale = np.log(df)
plt.title('Log of Dataset')
plt.plot(df_logScale)
plt.show()

# Trend remains the same, although the values on y-axis have changed

# Next, calculate moving average
plt.figure(figsize=(10, 6))
movingAverage = df_logScale.rolling(window=12).mean()
movingSTD = df_logScale.rolling(window=12).std()
plt.title('Log of Dataset with Moving Average')
plt.plot(df_logScale)
plt.plot(movingAverage, color='red')
plt.show()


# Test for stationarity
def test_stationarity(timeseries):
    #Determining rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    # Plot rolling statistics
    plt.figure(figsize=(10, 6))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show()

    # Determine Dickey-Fuller
    print("Results of Dickey-Fuller test")
    adft = adfuller(timeseries, autolag='AIC')

    # Output for dft will give the result without defining what the values are
    # Hence, we manually write what values it explains using a for loop
    output = pd.Series(adft[0:4], index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])

    for key, values in adft[4].items():
        output['critical value (%s)'%key] = values

    print(output)
    print()


# determine the difference between the moving average
# and the actual gas consumption
dfScaleMinueMovAvg = df_logScale - movingAverage
print('Original Moving Average')
print(dfScaleMinueMovAvg.head(15))
print()

# Remove NaN values
dfScaleMinueMovAvg.dropna(inplace=True)
print('Removed NAs')
print(dfScaleMinueMovAvg.head(15))
print()
test_stationarity(dfScaleMinueMovAvg)

# Calculate the weighted average of the time series to see the trend that is present
exponentialDecayWeightedAverage = df_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()

plt.figure(figsize=(10, 6))
plt.plot(df_logScale)
plt.plot(exponentialDecayWeightedAverage, color='red')
plt.title('Exponential Decay Weighted Average')
plt.show()

df_logScaleMinusMovingExponentialDecayAverage = df_logScale - exponentialDecayWeightedAverage
test_stationarity(df_logScaleMinusMovingExponentialDecayAverage)

# Shift the values into time series so that we can use it in forecasting
df_LogDiffShifting = df_logScale - df_logScale.shift()

plt.figure(figsize=(10, 6))
plt.plot(df_LogDiffShifting)
plt.title('Log Differential Shifting')
plt.show()

# Drop the NA values
df_LogDiffShifting.dropna(inplace=True)
test_stationarity(df_LogDiffShifting)

from statsmodels.tsa.seasonal import seasonal_decompose

# Seperate trend and seasonality from the time series using additive decomposition
decomposition = seasonal_decompose(df_logScale)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Visualize components
plt.figure(figsize=(10, 6))
plt.subplot(411)
plt.plot(df_logScale, label="Original")
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label="Trend")
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label="Seasonality")
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label="Residuals")
plt.legend(loc='best')
plt.tight_layout

# Use moving average smoothing method remove fluctuations from a transformed time series data
# Use a 3-months moving average
decomposedLogdata = residual
decomposedLogdata.dropna(inplace=True)
test_stationarity(decomposedLogdata)

# ACF and PACF plots
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(df_LogDiffShifting, nlags=20)
lag_pacf = pacf(df_LogDiffShifting, nlags=20, method='ols')

# ACF
plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_LogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_LogDiffShifting)), linestyle='--', color='gray')
plt.title("Autocorrelation Function")

# PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_LogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_LogDiffShifting)), linestyle='--', color='gray')
plt.title("PartialAutocorrelation Function")
plt.show()