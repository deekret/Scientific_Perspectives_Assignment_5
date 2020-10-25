# Load libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

data_2011 = pd.read_csv("pp_gas_emission/gt_2011.csv")
data_2012 = pd.read_csv("pp_gas_emission/gt_2012.csv")
data_2013 = pd.read_csv("pp_gas_emission/gt_2013.csv")
data_2014 = pd.read_csv("pp_gas_emission/gt_2014.csv")
data_2015 = pd.read_csv("pp_gas_emission/gt_2015.csv")

features1 = data_2011.columns[:9]
features2 = data_2011.columns[10:11]
features = features1.append(features2)

X_train = np.concatenate((data_2011.values[:, :9], data_2012.values[:, :9]))
y_train = np.concatenate((data_2011.values[:, -1], data_2012.values[:, -1]))

X_val = data_2013.values[:, :9]
y_val = data_2013.values[:, -1]

X_test = np.concatenate((data_2014.values[:, :9], data_2015.values[:, :9]))
y_test = np.concatenate((data_2014.values[:, -1], data_2015.values[:, -1]))

X_train_2 = np.concatenate((X_train, X_val))
y_train_2 = np.concatenate((y_train, y_val))

model1 = LinearRegression().fit(X_train, y_train)
model2 = LinearRegression().fit(X_train_2, y_train_2)

y_predict1 = model1.predict(X_val)
y_predict2 = model2.predict(X_test)
print(y_predict1)
print(y_predict2)

r_sq_1 = model1.score(X_val, y_val)
spcorr_1, p1 = spearmanr(y_val, y_predict1)

r_sq_2 = model2.score(X_test, y_test)
spcorr_2, p2 = spearmanr(y_test, y_predict2)

print(" MODEL 1")
print("mean absolute error:", mean_absolute_error(y_val, y_predict1))
print("spearman correlation:", spcorr_1)
print('R_squared:', r_sq_1)
print()
print(" MODEL 2")
print("mean absolute error:", mean_absolute_error(y_test, y_predict2))
print("spearman correlation:", spcorr_2)
print('R_squared:', r_sq_2)
print()






