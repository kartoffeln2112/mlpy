import pandas as pd
import seaborn as sns
import numpy as np
from statsmodels import tsa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("Reading Data")
moldi = pd.read_csv("GI_Wall2_Mould_P5_A8_C.csv")
print(moldi.shape)

# Clean Data
print("Cleaning Data")

# drop columns
moldi = moldi.drop(columns=["Unnamed: 4"])

# convert data types, replace invalid values
moldi["Mmax"].replace('#DIV/0!', "0", inplace=True)
moldi["Mmax"] = moldi["Mmax"].astype("float64")

moldi["k2"].replace('#DIV/0!', "0", inplace=True)
moldi["k2"] = moldi["k2"].astype("float64")

moldi["Growth"].replace('#NUM!', "0", inplace=True)
moldi["Growth"] = moldi["Growth"].astype("float64")

moldi["Growth.1"].replace('#NUM!', "0", inplace=True)
moldi["Growth.1"] = moldi["Growth.1"].astype("float64")

# moldi["Date"] = pd.to_datetime(moldi["Date"])

# add new column: Time At Risk
moldi["atRisk"] = 0
for i in range(moldi.shape[0]):
    if moldi.at[i, "Recession"] == 0:
        if begin:
            x = moldi.at[i, "Time"]
            begin = False
        moldi.at[i, "atRisk"] = moldi.at[i, "Time"] - x
        continue
    begin = True

print(moldi.head())
print(moldi.shape)
print(moldi.dtypes)

# plot
moldi = moldi.set_index('Time')
moldi.plot(linewidth=2, fontsize=12)
plt.show()

# create time series model


X = moldi.loc[:, ["Temp"]]
y = moldi.loc[:, ["RHcrit"]]
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=1234)

mim1LogR = LinearRegression().fit(trainX, trainY)
