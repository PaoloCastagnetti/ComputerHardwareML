# -*- coding: utf-8 -*-
"""
ML project
DataSet: Computer Hardware Data Set
link DS: https://archive.ics.uci.edu/ml/datasets/Computer+Hardware
Associated task: Regression

Attributes:
1. Vendor name
2. Model Name
3. MYCT: machine cycle time in nanoseconds (integer)
4. MMIN: minimum main memory in kilobytes (integer)
5. MMAX: maximum main memory in kilobytes (integer)
6. CACH: cache memory in kilobytes (integer)
7. CHMIN: minimum channels in units (integer)
8. CHMAX: maximum channels in units (integer)
9. PRP: published relative performance (integer)
10. ERP: estimated relative performance from the original article (integer)

@author: Paolo Castagnetti n. matr. 143098 267731@studenti.unimore.it
"""

import warnings
import pandas as pd
import seaborn as sns
from time import time
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.style.use('seaborn')
#Per ignorare alcune tipologie di warning
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = UserWarning)

#Recupero del dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data'
data = pd.read_csv(url, header = None)

#Definizione delle colonne
data.columns = ['VENDOR', 'MODEL', 'MYCT', 'MMIN', 
                 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']

Model_Compare = [[]]

#Drop delle colonne non utilizzate
n_columns = data.columns.drop(['VENDOR', 'PRP', 'MODEL', 'ERP'])
d_vendor = pd.get_dummies(data['VENDOR'], prefix = 'vnd', drop_first = True)

#Feature correlation
sns.heatmap(data[['MYCT', 'MMIN', 'MMAX', 'CACH','CHMIN','CHMAX', 'PRP']].corr(), annot = True)
plt.title('Correlation Matrix', fontsize = 14)


####################### Predictions with vendor feature ####################################

print('\n'+15*'-'+"Prediction with VENDOR"+15*'-')
#prepare train, test and validation data
y = data['PRP']

X = pd.concat([data[n_columns], d_vendor], axis = 1)

#Divisione in dataset di training, validation e testing
X_tr,X_ts,y_tr,y_ts=train_test_split(X,y,test_size = 0.1, 
                                     random_state = 42, shuffle = True)
X_tr, X_vl, y_tr, y_vl= train_test_split(X_tr, y_tr,
                                         test_size = 0.2, random_state = 42)

#Scalamento delle feature
scaler = StandardScaler()
y_scaler = StandardScaler()

y_tr = y_tr.values.reshape(-1, 1)
y_vl = y_vl.values.reshape(-1, 1)
y_ts = y_ts.values.reshape(-1, 1)

#Fit sul training set
scaler.fit(X_tr)
y_scaler.fit(y_tr)

#Trasformazione di training e testing set
X_tr = scaler.transform(X_tr)
X_vl = scaler.transform(X_vl)
X_ts = scaler.transform(X_ts)
y_tr = y_scaler.transform(y_tr)
y_vl = y_scaler.transform(y_vl)
y_ts = y_scaler.transform(y_ts)

####################### Linear Regression Model w/Vendor ##############################################

#Definizione del modello
lin_reg = LinearRegression()

#Fitting dei dati sul modello
sta_fit = time()
lin_reg.fit(X_tr,y_tr)
sto_fit = time()
Fitting_time = (sto_fit - sta_fit)* 1000

#Predizione sul set di validation
Y_pred = lin_reg.predict(X_vl)

#Validation Score
_r2_score = r2_score(y_vl,Y_pred)
MSE = mean_squared_error(y_vl,Y_pred)
MAE = mean_absolute_error(y_vl, Y_pred)

print('\n'+16*'-'+"Linear Regression:"+16*'-'+'\n')
print("Fitting time: ", Fitting_time, ' ms')
print("R^2 score for lin_reg validation set: %.4f"%_r2_score)
print("Mean Square Error for Validation set: %.5f"%MSE)
print("Mean Absolute Error for Validation set: %.5f"%MAE)
print('\n')

Model_Compare[0] = ['Linear Regression', round(_r2_score, 4), round(MSE,5), round(MAE,5), round(Fitting_time, 5)]

####################### Lasso Model w/Vendor ##############################################

print('\n'+16*'-'+"Lasso Regression:"+16*'-')

#Definizione dei possibili alpha
alphas = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0]
init_r2 = 0

print("\nLasso Regression: Best Parameters")
sta_fit = time()

#Ricerca dei parametri migliori per il modello
for i in alphas:
    lasso_reg_tmp = Lasso(alpha = i, max_iter = 5000)
    lasso_reg_tmp.fit(X_tr, y_tr)
    tr_r2 = r2_score(y_tr, lasso_reg_tmp.predict(X_tr)) # train
    vl_r2 = r2_score(y_vl, lasso_reg_tmp.predict(X_vl)) # validation
    if vl_r2 > init_r2:
        print("Aplha: ", i, "\nTrain R-Squared value:", round(tr_r2, 5),
             "\nValidation R-squared value:", round(vl_r2, 5))
        init_r2 = vl_r2
        lasso_reg = lasso_reg_tmp
        alpha_lasso=i

sto_fit = time()
Fitting_time = (sto_fit - sta_fit)* 1000

#Predizione sul set di validation
Y_pred = lasso_reg.predict(X_vl)

#Validation Score
lasso_r2_score = r2_score(y_vl,Y_pred)
MSE = mean_squared_error(y_vl,Y_pred)
MAE = mean_absolute_error(y_vl, Y_pred)

print("\nFitting time: ", Fitting_time, ' ms')
print("R^2 score for lasso_reg validation set: %.4f"%lasso_r2_score)
print("Mean Square Error for Validation set: %.5f"%MSE)
print("Mean Absolute Error for Validation set: %.5f"%MAE)
print('\n')
Model_Compare.append(['Lasso Regression', round(lasso_r2_score, 4), round(MSE,5), round(MAE,5), round(Fitting_time, 5)])

####################### Ridge Model w/Vendor ##############################################

print('\n'+16*'-'+"Ridge Regression:"+16*'-')

#Definizione dei possibili alpha
alphas = [1e-4, 1e-3, 1e-2, -.1, -.5, 1.0, 5.0, 10.0]
init_r2 = 0

print('\nRidge Regression: Best Parameters')
sta_fit = time()

#Ricerca dei parametri migliori per il modello
for i in alphas:
    Ridge_reg_tmp = Ridge(alpha = i)
    Ridge_reg_tmp.fit(X_tr, y_tr)
    tr_r2 = r2_score(y_tr, Ridge_reg_tmp.predict(X_tr)) # train
    vl_r2 = r2_score(y_vl, Ridge_reg_tmp.predict(X_vl)) # validation
    if vl_r2 > init_r2:
        print("Alpha: ", i, "\nTrain R-Squared value:", round(tr_r2, 5),
             "\nValidation R-squared value:", round(vl_r2, 5))
        init_r2 = vl_r2
        Ridge_reg = Ridge_reg_tmp
        alpha_ridge = i

sto_fit = time()
Fitting_time = (sto_fit - sta_fit)* 1000

#Predizione sul set di validation
Y_pred = Ridge_reg.predict(X_vl)

#Validation Score
ridge_r2_score = r2_score(y_vl,Y_pred)
MSE = mean_squared_error(y_vl,Y_pred)
MAE = mean_absolute_error(y_vl, Y_pred)

print("\nFitting time: ", Fitting_time, ' ms')
print("R^2 score for Ridge_reg validation set: %.4f"%ridge_r2_score)
print("Mean Square Error for Validation set: %.5f"%MSE)
print("Mean Absolute Error for Validation set: %.5f"%MAE)


Model_Compare.append(['Ridge Regression', round(ridge_r2_score, 4), round(MSE,5), round(MAE,5), round(Fitting_time, 5)])

####################### Printing models comparision table w/Vendor ####################################

print('\n'+7*'-'+'Model || With Vendors Data || Result:'+7*'-')
Comparision_Table = DataFrame(Model_Compare, columns = ['Model','R2 Score','MSE', 'MAE', 'Fitting time'])
print(Comparision_Table)