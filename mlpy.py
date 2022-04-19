import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics

from statsmodels import tsa
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

import pickle


print("Reading Data")
moldi = pd.read_csv("GI_Wall2_Mould_P5_A8_C.csv")
moldii = pd.read_csv("X_Wall2_Mould_P6_B3_C.csv")
print("Moldi")
print(moldi.head())
print(moldi.shape)

print("Moldii")
print(moldii.head())
print(moldii.shape)

# Clean Data
def cleanMoldDF(molddf):
    # Clean Data
    print("Cleaning Data")

    # drop columns
    molddf = molddf.drop(columns=["Unnamed: 3"])

    # convert data types, replace invalid values
    molddf["Mmax"].replace('#DIV/0!', "0", inplace=True)
    molddf["Mmax"] = molddf["Mmax"].astype("float64")

    molddf["k2"].replace('#DIV/0!', "0", inplace=True)
    molddf["k2"] = molddf["k2"].astype("float64")

    molddf["Growth"].replace('#NUM!', "0", inplace=True)
    molddf["Growth"] = molddf["Growth"].astype("float64")

    molddf["Growth.1"].replace('#NUM!', "0", inplace=True)
    molddf["Growth.1"] = molddf["Growth.1"].astype("float64")

    # moldi["Date"] = pd.to_datetime(moldi["Date"])

    # add new column: Time At Risk
    molddf["atRisk"] = 0
    for i in range(molddf.shape[0]):
        if molddf.at[i, "Recession"] == 0:
            if begin:
                x = molddf.at[i, "Time"]
                begin = False
            molddf.at[i, "atRisk"] = molddf.at[i, "Time"] - x
            continue
        begin = True

    # add new column: M.1-1
    molddf["RHdiff"] = [(molddf.at[r, "RH"] - molddf.at[r, "RHcrit"]) if r != (molddf.shape[0] - 1) else 0
                        for r in range(molddf.shape[0])]

    # set index column as Time
    molddf = molddf.set_index('Time')

    print(molddf.head())
    print(molddf.shape)
    print(molddf.dtypes)
    return molddf


moldi = cleanMoldDF(moldi)
moldii = cleanMoldDF(moldii)

# plot
"""moldi.loc[:, ["Temp", "RH", "RHcrit", "Growth", "Growth.1", "Recession", "M", "M.1"]].plot(linewidth=2, fontsize=12)
plt.show()

moldii.loc[:, ["Temp", "RH", "RHcrit", "Growth", "Growth.1", "Recession", "M", "M.1"]].plot(linewidth=2, fontsize=12)
plt.show()

moldi.loc[:, ["Mmax", "M.1"]].plot()
plt.show()

moldi.loc[:, ["Growth", "M"]].plot()
plt.show()

moldi.describe()
moldii.describe()

# create time series model
# from statsmodels.graphics.tsaplots import plot_acf,plot_pacfs
mendog = moldi.loc[:, ["RHcrit"]]
m2endog = moldi.loc[:, ["RHcrit"]]

mexog = moldii.loc[:, ["Temp"]]
m2exog = moldii.loc[:, ["Temp"]]

mArima = ARIMA(mendog, exog = mexog)
mResArima = mArima.fit()
print(mResArima.summary())

m2Arima = ARIMA(m2endog, exog = m2exog)
m2ResArima = m2Arima.fit()
print(m2ResArima.summary())

plt.plot(moldi.loc[:, ["RHcrit"]], color="lightpink")
plt.plot(mResArima.fittedvalues, color="red")
plt.show()
plt.plot(moldii.loc[:, ["RHcrit"]], color="lightblue")
plt.plot(m2ResArima.fittedvalues, color="blue")"""

# correlation matrix
miCorrMatrix = moldi.corr().abs()
miiCorrMatrix = moldii.corr().abs()

print("Moldi")
sns.heatmap(miCorrMatrix.loc[:, ["Mmax", "Growth", "Growth.1", "M", "M.1"]], annot=True)
plt.show()

# print("Moldii")
# sns.heatmap(miiCorrMatrix.loc[:, ["Mmax", "Growth", "Growth.1", "M", "M.1"]], annot=True)
# plt.show()

# MMAX MODEL
X1 = moldi.loc[1:, ["Temp", "RH", "RHdiff"]]
y = moldi.loc[1:, "Mmax"]
trainX, testX, trainY, testY = train_test_split(X1, y, test_size=0.2, random_state=1234)

mimmLogR = LinearRegression().fit(trainX, trainY)
predmimmLR = mimmLogR.predict(testX)
mse = np.mean((predmimmLR - testY)**2)
print("Intercept: ", mimmLogR.intercept_)
print("Coefs: ", mimmLogR.coef_)
print("Score: ", mimmLogR.score(trainX, trainY))
print("R2: ", metrics.r2_score(testY, predmimmLR))
print("MSE: ", mse)

# M.growth MODEL
XwM = moldi.loc[1:, ["Growth", "Recession"]]
predXmimmLR = mimmLogR.predict(X1)
XwM["Mmax"] = predXmimmLR

y = moldi.loc[1:, "M"]
trainX, testX, trainY, testY = train_test_split(XwM, y, test_size=0.2, random_state=1234)

mimwmLogR = LinearRegression().fit(trainX, trainY)
predmimwmLR = mimwmLogR.predict(testX)
mse = np.mean((predmimwmLR - testY)**2)
print("Intercept: ", mimwmLogR.intercept_)
print("Coefs: ", mimwmLogR.coef_)
print("Score: ", mimwmLogR.score(trainX, trainY))
print("R2: ", metrics.r2_score(testY, predmimwmLR))
print("MSE: ", mse)


with open('model.pkl', 'wb') as mmax_model_file:
    pickle.dump(mimmLogR, mmax_model_file)

with open('model.pkl', 'wb') as m_model_file:
    pickle.dump(mimwmLogR, m_model_file)
