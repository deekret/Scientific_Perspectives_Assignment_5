# Load libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
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
#get all the files for the model
trainFiles = []
trainFiles.append(data_2011)
trainFiles.append(data_2012)
frame = pd.concat(trainFiles, axis=0, ignore_index=True)

X_train = frame[['AT','AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']]
y_train = frame[['NOX']]

validationFrame = data_2013
X_val = validationFrame[['AT','AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']]
y_val = validationFrame[['NOX']]

testFiles = []
testFiles.append(data_2014)
testFiles.append(data_2015)
frame2 = pd.concat(testFiles, axis=0, ignore_index=True)
X_test = frame2[['AT','AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']]
y_test = frame2[['NOX']]

#normalize the data
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
print()
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

X_OptimizationTrain = frame[['AT','AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']]  #'AT','AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP'
Y_OptimizationTrain = frame[['NOX']]
optimizationTrainDf = pd.DataFrame(X_OptimizationTrain)

CDPXTITtrain = []
# TEYXTITtrain = []
# CDPXTEYtrain = []
TEYXATtrain = []
AP2train = []
TEYXAFDPtrain = []
# ATXCDPtrain = []
AFDPminAPtrain = []
# AFDP2train = []
# TITXTATtrain = []
# TEYsqrttrain = []
# ATXAPtrain = []
i = 0
while i < len(frame):
    CDPXTITtrain.append(optimizationTrainDf['CDP'][i] * optimizationTrainDf['TIT'][i])
    # TEYXTITtrain.append(optimizationTrainDf['TEY'][i] * optimizationTrainDf['TIT'][i])
    # CDPXTEYtrain.append(optimizationTrainDf['CDP'][i] * optimizationTrainDf['TEY'][i])
    TEYXATtrain.append(optimizationTrainDf['TEY'][i] * optimizationTrainDf['AT'][i])
    AP2train.append(optimizationTrainDf['AP'][i] * optimizationTrainDf['AP'][i])
    TEYXAFDPtrain.append(optimizationTrainDf['TEY'][i] * optimizationTrainDf['AFDP'][i])
    # ATXCDPtrain.append(optimizationTrainDf['AT'][i] * optimizationTrainDf['CDP'][i])
    # AFDPminAPtrain.append(optimizationTrainDf['AFDP'][i] / optimizationTrainDf['AP'][i])
    # AFDP2train.append(optimizationTrainDf['AFDP'][i] * optimizationTrainDf['AFDP'][i])
    # TITXTATtrain.append(optimizationTrainDf['TIT'][i] * optimizationTrainDf['TAT'][i])
    # TEYsqrttrain.append(math.sqrt(optimizationTrainDf['TEY'][i]))
    # ATXAPtrain.append(optimizationTrainDf['AT'][i] * optimizationTrainDf['AP'][i])
    i = i + 1
optimizationTrainDf['CDPXTIT'] = CDPXTITtrain
# optimizationTrainDf['CDPXTEY'] = CDPXTEYtrain
# optimizationTrainDf['TEYXTIT'] = TEYXTITtrain
optimizationTrainDf['TEYXAT'] = TEYXATtrain
optimizationTrainDf['AP2'] = AP2train
optimizationTrainDf['TEYXAFDP'] = TEYXAFDPtrain
# optimizationTrainDf['ATXCDP'] = ATXCDPtrain
# optimizationTrainDf['AFDPminAP'] = AFDPminAPtrain
# optimizationTrainDf['AFDP2'] = AFDP2train
# optimizationTrainDf['TITXTAT'] = TITXTATtrain
# optimizationTrainDf['TEYsqrt'] = TEYsqrttrain
# optimizationTrainDf['ATXAP'] = ATXAPtrain

validationFrame = data_2013
X_OptimizationVal = validationFrame[['AT','AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']]  #'AT','AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP'
Y_OptimizationVal = validationFrame[['NOX']]
optimizationValDf = pd.DataFrame(X_OptimizationVal)

CDPXTITval = []
# TEYXTITval = []
# CDPXTEYval = []
TEYXATval = []
AP2val = []
TEYXAFDPval = []
# ATXCDPval = []
# AFDPminAPval = []
# AFDP2val = []
# TITXTATval = []
# TEYsqrtval = []
# ATXAPval = []
x = 0
while x < len(validationFrame):
    CDPXTITval.append(optimizationValDf['CDP'][x] * optimizationValDf['TIT'][x])
    # TEYXTITval.append(optimizationValDf['TEY'][x] * optimizationValDf['TIT'][x])
    # CDPXTEYval.append(optimizationValDf['CDP'][x] * optimizationValDf['TEY'][x])
    TEYXATval.append(optimizationValDf['TEY'][x] * optimizationValDf['AT'][x])
    AP2val.append(optimizationValDf['AP'][x] * optimizationValDf['AP'][x])
    TEYXAFDPval.append(optimizationValDf['TEY'][x] * optimizationValDf['AFDP'][x])
    # ATXCDPval.append(optimizationValDf['AT'][x] * optimizationValDf['CDP'][x])
    # AFDPminAPval.append(optimizationValDf['AFDP'][x] * optimizationValDf['AP'][x])
    # AFDP2val.append(optimizationValDf['AFDP'][x] * optimizationValDf['AFDP'][x])
    # TITXTATval.append(optimizationValDf['TIT'][x] * optimizationValDf['TAT'][x])
    # TEYsqrtval.append(math.sqrt(optimizationValDf['TEY'][x]))
    # ATXAPval.append(optimizationValDf['AT'][x] * optimizationValDf['AP'][x])
    x = x + 1
optimizationValDf['CDPXTIT'] = CDPXTITval
# optimizationValDf['CDPXTEY'] = CDPXTEYval
# optimizationValDf['TEYXTIT'] = TEYXTITval
optimizationValDf['TEYXAT'] = TEYXATval
optimizationValDf['AP2'] = AP2val
optimizationValDf['TEYXAFDP'] = TEYXAFDPval
# optimizationValDf['ATXCDP'] = ATXCDPval
# optimizationValDf['AFDPminAP'] = AFDPminAPval
# optimizationValDf['AFDP2'] = AFDP2val
# optimizationValDf['TITXTAT'] = TITXTATval
# optimizationValDf['TEYsqrt'] = TEYsqrtval
# optimizationValDf['ATXAP'] = ATXAPval

files2 = []
files2.append(data_2014)
files2.append(data_2015)
frame2 = pd.concat(files2, axis=0, ignore_index=True)
X_OptimizationTest = frame2[['AT','AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']]  #'AT','AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP'
Y_OptimizationTest = frame2[['NOX']]
optimizationTestDf = pd.DataFrame(X_OptimizationTest)

CDPXTITtest = []
TEYXATtest  = []
AP2test  = []
TEYXAFDPtest = []
x = 0
while x < len(frame2):
    CDPXTITtest.append(optimizationTestDf['CDP'][x] * optimizationTestDf['TIT'][x])
    TEYXATtest.append(optimizationTestDf['TEY'][x] * optimizationTestDf['AT'][x])
    AP2test.append(optimizationTestDf['AP'][x] * optimizationTestDf['AP'][x])
    TEYXAFDPtest.append(optimizationTestDf['TEY'][x] * optimizationTestDf['AFDP'][x])
    x = x + 1
optimizationTestDf['CDPXTIT'] = CDPXTITtest
optimizationTestDf['TEYXAT'] = TEYXATtest
optimizationTestDf['AP2'] = AP2test
optimizationTestDf['TEYXAFDP'] = TEYXAFDPtest

#Do a z normalisation with stat zscore
X_OptimizationTrainNorm = stats.zscore(optimizationTrainDf)
Y_OptimizationTrainNorm = stats.zscore(Y_OptimizationTrain)

X_OptimizationValNorm = stats.zscore(optimizationValDf)
Y_OptimizationValNorm = stats.zscore(Y_OptimizationVal)

X_OptimizationTestNorm = stats.zscore(X_OptimizationTest)
Y_OptimizationTestNorm = stats.zscore(Y_OptimizationTest)

#Do the linear regression with X and y
model3 = LinearRegression().fit(X_OptimizationTrainNorm, Y_OptimizationTrainNorm)

# #Create a prediction for y
y_predict3 = model3.predict(X_OptimizationValNorm)

spcorr_3_val, p3 = spearmanr(Y_OptimizationValNorm, y_predict3)
y_predict3 = model3.predict(X_OptimizationTestNorm)

# # Get spreaman correlation
# spcorr_3, p3 = spearmanr(Y_OptimizationValNorm, y_predict3)
# print(spcorr_3)
spcorr_3, p3 = spearmanr(Y_OptimizationTestNorm, y_predict3)
r_sq_3 = model3.score(X_OptimizationTestNorm, Y_OptimizationTestNorm)
mean_absolute_error_testing = mean_absolute_error(Y_OptimizationTestNorm, y_predict3)

print("mean absolute error:", mean_absolute_error_testing)
print("spearman correlation:", spcorr_3)
print('R_squared:', r_sq_3)

#place labels on the bars
importance = model3.coef_[0]
importances = []
sum = 0
for x in importance:
    sum += abs(x)
for x in importance:
    x = (abs(x) / sum)
    importances.append(x)
# Engineered features F1 = 'CDPXTIT', F2 = 'TEYXAT', F3 = 'AP2', F4 = 'TEYXAFDP'
plt.bar(['AT','AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'F1', 'F2', 'F3', 'F4'], importances)
plt.show()

print("####################")
print("Instance at position 0 of test set: ")
print(X_OptimizationValNorm[0])
print(importance)

y = 0
for x,w in zip(X_OptimizationTestNorm[0], importance):
    y += w*x
print("calculated value:", y)
print("actually generated value:", y_predict3[0])
print()

print("####################")
print("Instance at position 200 of test set: ")
print(X_OptimizationValNorm[200])
print(importance)

y = 0
for x,w in zip(X_OptimizationTestNorm[199], importance):
    y += w*x
print("calculated value:", y)
print("actually generated value:", y_predict3[199])
print()


print("#################")

print("#################")
print("Phase 3:")
validationSplit = data_2013
X_SplitVal = validationSplit[['AT','AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']]  #'AT','AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP'
Y_SplitVal = validationSplit[['NOX']]

files3 = []
files3.append(data_2014)
files3.append(data_2015)
testingSplit = pd.concat(files3, axis=0, ignore_index=True)
X_SplitTest = testingSplit[['AT','AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP']]  #'AT','AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP'
Y_SplitTest = testingSplit[['NOX']]

X_SplitValNorm = stats.zscore(X_SplitVal)
Y_SplitValNorm = stats.zscore(Y_SplitVal)

X_SplitTestNorm = stats.zscore(X_SplitTest)
Y_SplitTestNorm = stats.zscore(Y_SplitTest)

split_model1_validation_array_x = np.array_split(X_SplitValNorm, 10)
split_model1_validation_array_y = np.array_split(Y_SplitValNorm, 10)

split_model3_validation_array_x = np.array_split(X_OptimizationValNorm, 10)
split_model3_validation_array_y = np.array_split(Y_OptimizationValNorm, 10)

split_model1_testing_array_x = np.array_split(X_SplitTestNorm, 20)
split_model1_testing_array_y = np.array_split(Y_SplitTestNorm, 20)

split_model3_testing_array_x = np.array_split(X_OptimizationTestNorm, 20)
split_model3_testing_array_y = np.array_split(Y_OptimizationTestNorm, 20)

#online learning with validation set and with model 1
val_model_1 = model1
x_model1_array = X_train
y_model1_array = y_train
y = 0
while y < 10:
    x_split = split_model1_validation_array_x[y]
    y_split = split_model1_validation_array_y[y]
    y_predictSplit = val_model_1.predict(x_split)
    spcorr_split, p3 = spearmanr(y_split, y_predictSplit)
    r_sq_split = val_model_1.score(x_split, y_split)
    mean_absolute_error_split = mean_absolute_error(y_split, y_predictSplit)
    print()
    print("training original set", y + 1)
    print("mean absolute error:", mean_absolute_error_split)
    print("spearman correlation:", spcorr_split)
    print('R_squared:', r_sq_split)
    print()

    x_model1_array = np.concatenate((x_model1_array, x_split))
    y_model1_array = np.concatenate((y_model1_array, y_split))

    val_model_1 = LinearRegression().fit(x_model1_array, y_model1_array)
    y = y + 1

#online learning with validation set and with our own created features
val_model_3 = model3
x_model3_array = X_OptimizationTrainNorm
y_model3_array = Y_OptimizationTrainNorm
q = 0
while q < 10:
    x_split = split_model3_validation_array_x[q]
    y_split = split_model3_validation_array_y[q]
    y_predictSplit = val_model_3.predict(x_split)
    spcorr_split, p3 = spearmanr(y_split, y_predictSplit)
    r_sq_split = val_model_3.score(x_split, y_split)
    mean_absolute_error_split = mean_absolute_error(y_split, y_predictSplit)
    print()
    print("training created feature set", q + 1)
    print("mean absolute error:", mean_absolute_error_split)
    print("spearman correlation:", spcorr_split)
    print('R_squared:', r_sq_split)
    print()

    x_model3_array = np.concatenate((x_model3_array, x_split))
    y_model3_array = np.concatenate((y_model3_array, y_split))

    val_model_3 = LinearRegression().fit(x_model3_array, y_model3_array)
    q = q + 1

#============================================================================================
# Perform a statistical test to compare the original set of features and our own engineered features

y_predict1 = model1.predict(x_model1_array)
y_predict3 = model3.predict(x_model3_array)

spcorr_split1, p = spearmanr(y_model1_array, y_predict1)
spcorr_split3, p = spearmanr(y_model3_array, y_predict3)

print("Spearman correlation of model 1 prediction: ", spcorr_split1)
print("Spearman correlation of model 3 prediction: ", spcorr_split3)

stat, p = ttest_rel(y_predict1, y_predict3)
print("TTest between original feature prediction and engineered feature prediction: ", stat)

#y_predict_test = model3.predict((X_OptimizationTestNorm))
#spcorr_test, p = spearmanr(y_predict3, Y_OptimizationTestNorm)
print("Spearman correlation of model 3 prediction: ", spcorr_split3)
print("Spearman correlation of phase 2 model prediction: ", spcorr_3_val)

#============================================================================================
# online learning with testing set and with model 1
test_model_1 = model1
x_model1_array = np.concatenate((X_train, X_SplitValNorm))
y_model1_array = np.concatenate((y_train, Y_SplitValNorm))
y = 0
while y < 20:
    x_split = split_model1_testing_array_x[y]
    y_split = split_model1_testing_array_y[y]
    y_predictSplit = test_model_1.predict(x_split)
    spcorr_split, p3 = spearmanr(y_split, y_predictSplit)
    r_sq_split = test_model_1.score(x_split, y_split)
    mean_absolute_error_split = mean_absolute_error(y_split, y_predictSplit)
    print()
    print("training original set", y + 1)
    print("mean absolute error:", mean_absolute_error_split)
    print("spearman correlation:", spcorr_split)
    print('R_squared:', r_sq_split)
    print()

    x_model1_array = np.concatenate((x_model1_array, x_split))
    y_model1_array = np.concatenate((y_model1_array, y_split))

    test_model_1 = LinearRegression().fit(x_model1_array, y_model1_array)
    y = y + 1

#online learning with testing set and with our own created features
test_model_3 = model3
x_model3_array = np.concatenate((X_OptimizationTrainNorm, X_OptimizationValNorm))
y_model3_array = np.concatenate((Y_OptimizationTrainNorm, Y_OptimizationValNorm))
q = 0
while q < 20:
    x_split = split_model3_testing_array_x[q]
    y_split = split_model3_testing_array_y[q]
    y_predictSplit = test_model_3.predict(x_split)
    spcorr_split, p3 = spearmanr(y_split, y_predictSplit)
    r_sq_split = test_model_3.score(x_split, y_split)
    mean_absolute_error_split = mean_absolute_error(y_split, y_predictSplit)
    print()
    print("training created feature set", q + 1)
    print("mean absolute error:", mean_absolute_error_split)
    print("spearman correlation:", spcorr_split)
    print('R_squared:', r_sq_split)
    print()

    x_model3_array = np.concatenate((x_model3_array, x_split))
    y_model3_array = np.concatenate((y_model3_array, y_split))

    test_model_3 = LinearRegression().fit(x_model3_array, y_model3_array)
    q = q + 1
print("#################")

#============================================================================================
# Perform a statistical test to compare the original set of features and our own engineered features

y_predict1 = test_model_1.predict(x_model1_array)
y_predict3 = test_model_3.predict(x_model3_array)

spcorr_split1, p = spearmanr(y_model1_array, y_predict1)
spcorr_split3, p = spearmanr(y_model3_array, y_predict3)

print("Spearman correlation of model 1 prediction: ", spcorr_split1)
print("Spearman correlation of model 3 prediction: ", spcorr_split3)

stat, p = ttest_rel(y_predict1, y_predict3)
print("TTest between original feature prediction and engineered feature prediction: ", stat)

#y_predict_test = model3.predict((X_OptimizationTestNorm))
#spcorr_test, p = spearmanr(y_predict3, Y_OptimizationTestNorm)
print("Spearman correlation of model 3 prediction: ", spcorr_split3)
print("Spearman correlation of phase 2 model prediction: ", spcorr_3)

#============================================================================================









