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
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
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
#prepare train, and test data
y = data['PRP']

X = pd.concat([data[n_columns], d_vendor], axis = 1)

#Divisione in dataset di training e testing
X_tr,X_ts,y_tr,y_ts=train_test_split(X,y,test_size = 0.15, 
                                     random_state = 42, shuffle = True)

#Scalamento delle feature
scaler = StandardScaler()
y_scaler = StandardScaler()

y_tr = y_tr.values.reshape(-1, 1)
y_ts = y_ts.values.reshape(-1, 1)

#Fit sul training set
scaler.fit(X_tr)
y_scaler.fit(y_tr)

#Trasformazione di training e testing set
X_tr = scaler.transform(X_tr)
X_ts = scaler.transform(X_ts)
y_tr = y_scaler.transform(y_tr)
y_ts = y_scaler.transform(y_ts)

####################### Lasso Model w/Vendor ##############################################

print('\n'+16*'-'+"Lasso Regression:"+16*'-')
alpha = 0.0001

#Definizione del modello
lasso_reg = Lasso(alpha = alpha, max_iter = 5000)

#Fitting dei dati sul modello
sta_fit = time()
lasso_reg.fit(X_tr, y_tr)
sto_fit = time()

Fitting_time = (sto_fit - sta_fit)* 1000

#Predizione sul set di test
Y_pred = lasso_reg.predict(X_ts)

#Testing Score
lasso_r2_score = r2_score(y_ts,Y_pred)
MSE = mean_squared_error(y_ts,Y_pred)
MAE = mean_absolute_error(y_ts, Y_pred)

print("\nFitting time: ", Fitting_time, ' ms')
print("R^2 score for lasso_reg testing set: %.4f"%lasso_r2_score)
print("Mean Square Error for testing set: %.5f"%MSE)
print("Mean Absolute Error for testing set: %.5f"%MAE)
print('\n')
