# Load libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data_2011 = pd.read_csv("pp_gas_emission/gt_2011.csv")
data_2012 = pd.read_csv("pp_gas_emission/gt_2012.csv")
data_2013 = pd.read_csv("pp_gas_emission/gt_2013.csv")
data_2014 = pd.read_csv("pp_gas_emission/gt_2014.csv")
data_2015 = pd.read_csv("pp_gas_emission/gt_2015.csv")

features1 = data_2011.columns[:9]
features2 = data_2011.columns[10:11]
features = features1.append(features2)

print("#################")
print("Phase 1:")
print("#################")

X_train = np.concatenate((data_2011.values[:, :9], data_2012.values[:, :9]))
y_train = np.concatenate((data_2011.values[:, -1], data_2012.values[:, -1]))

X_val = data_2013.values[:, :9]
y_val = data_2013.values[:, -1]

X_test = np.concatenate((data_2014.values[:, :9], data_2015.values[:, :9]))
y_test = np.concatenate((data_2014.values[:, -1], data_2015.values[:, -1]))

X_train = stats.zscore(X_train)
y_train = stats.zscore(y_train)
X_test = stats.zscore(X_test)
y_test = stats.zscore(y_test)
X_val = stats.zscore(X_val)
y_val = stats.zscore(y_val)

X_train_2 = np.concatenate((X_train, X_val))
y_train_2 = np.concatenate((y_train, y_val))

model1 = LinearRegression().fit(X_train, y_train)
model2 = LinearRegression().fit(X_train_2, y_train_2)

y_predict1 = model1.predict(X_val)
y_predict2 = model2.predict(X_test)

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

print("#################")
print("Phase 2:")
#Get the training data and combine them
files = []
files.append(data_2011)
files.append(data_2012)
frame = pd.concat(files, axis=0, ignore_index=True)

X_OptimizationTrain = frame[['AT','AP', 'AFDP', 'TIT', 'TAT', 'TEY', 'CDP']]  #'AT','AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP'
Y_OptimizationTrain = frame[['NOX']]
optimizationTrainDf = pd.DataFrame(X_OptimizationTrain)

CDPXTITtrain = []
i = 0
while i < len(frame):
    CDPXTITtrain.append(optimizationTrainDf['CDP'][i] * optimizationTrainDf['TIT'][i])
    i = i + 1
optimizationTrainDf['CDPXTIT'] = CDPXTITtrain

validationFrame = data_2013
X_OptimizationVal = validationFrame[['AT','AP',  'AFDP', 'TIT', 'TAT', 'TEY', 'CDP']]  #'AT','AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP'
Y_OptimizationVal = validationFrame[['NOX']]
optimizationValDf = pd.DataFrame(X_OptimizationVal)

CDPXTITval = []
x = 0
while x < len(validationFrame):
    CDPXTITval.append(optimizationValDf['CDP'][x] * optimizationValDf['TIT'][x])
    x = x + 1
optimizationValDf['CDPXTIT'] = CDPXTITval

#Do a z normalisation with stat zscore
X_OptimizationTrainNorm = stats.zscore(optimizationTrainDf)
Y_OptimizationTrainNorm = stats.zscore(Y_OptimizationTrain)

X_OptimizationValNorm = stats.zscore(optimizationValDf)
Y_OptimizationValNorm = stats.zscore(Y_OptimizationVal)

# #Do the linear regression with X and y
model3 = LinearRegression().fit(X_OptimizationTrainNorm, Y_OptimizationTrainNorm)

# #Create a prediction for y
y_predict3 = model3.predict(X_OptimizationValNorm)

# # Get spreaman correlation
spcorr_3, p3 = spearmanr(Y_OptimizationValNorm, y_predict3)
print(spcorr_3)
# CDPXTIT = []
# i = 0
#
# while i < len(frame):
#     CDPXTIT.append(df['CDP'][i] * df['TIT'][i])
#     i = i + 1
# df['CDPXTIT'] = CDPXTIT

# OptimizationModel = LinearRegression().fit(X_OptimizationTrain, Y_OptimizationTrain)
# OptimizationModel = LinearRegression().fit(X_OptimizationTrain, Y_OptimizationTrain)
# y_predict3 = OptimizationModel.predict(X)
# spcorr_3, p1 = spearmanr(y, y_predict3)
# print("spearman correlation:", spcorr_3)
print("#################")







