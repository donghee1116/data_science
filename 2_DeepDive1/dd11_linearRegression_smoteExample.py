import os
from os.path import join
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

data = pd.read_csv('../data/creditcard.csv')
#print(data.head())
data = data.drop(['Time', 'Amount'], axis = 1)
sdscaler = StandardScaler()

# x는 독립변수 y는 종속변수
X = np.array(data.ix[:, data.columns != 'Class'])
y = np.array(data.ix[:, data.columns == 'Class'])

#데이터 train, test 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sm = SMOTE(random_state=2)
X_train_res , y_train_res = sm.fit_sample(X_train, y_train.ravel())

print("\nAfter OverSampling, the shape of X_train: \n", format(X_train_res.shape))

print("\nAfter OverSampling, the shape of y_train: \n", format(y_train_res.shape))

print("\nAfter OverSampling, counts of y_train_res '1' : \n", format(sum(y_train_res==1)))

print("\nAfter OverSampling, counts of y_train_res '0' : \n", format(sum(y_train_res==0)))

print("\n After Oversampling, the shape of X_test: \n", format(X_test.shape))

print("\n After Oversampling, the shape of y_test: \n", format(y_test.shape))

print("\n Before OverSampling, counts of label '1': \n", format(sum(y_test ==1)))

print("\n Before OverSampling, counts of label '0': \n", format(sum(y_test ==0)))


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()

lr_res = LogisticRegression()

# resampled model
lr_res.fit(X_train_res, y_train_res.ravel())

# resampled data
y_train_pre = lr_res.predict(X_train_res)
cnf_matrix_tra = confusion_matrix(y_train_res, y_train_pre)

print("\n(SMOTE + Standard) Train Recall : \n", cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])*100)

print("\n(SMOTE + Standard) Train Accuracy : \n", (cnf_matrix_tra[1,1] + cnf_matrix_tra[0,0])/ (cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1] +cnf_matrix_tra[0,1]+cnf_matrix_tra[0,0])*100)


y_pre = lr_res.predict(X_test)  # 새로운 데이터 검증
cnf_matrix_test = confusion_matrix(y_test, y_pre)

print('\n(Standard) test Recall: \n', cnf_matrix_test[1,1]/(cnf_matrix_test[1,0]+cnf_matrix_test[1,1])*100)

print('\n(Standard) test Accuracy: \n', (cnf_matrix_test[1,1] + cnf_matrix_test[0,0])/ (cnf_matrix_test[1,0]+cnf_matrix_test[1,1] +cnf_matrix_test[0,1]+cnf_matrix_test[0,0])*100)

print("\ncnf_matrix_test[0,0] >= \n",cnf_matrix_test[0,0])
print("\ncnf_matrix_test[0,1] >= \n",cnf_matrix_test[0,1])
print("\ncnf_matrix_test[1,0] >= \n",cnf_matrix_test[1,0])
print("\ncnf_matrix_test[1,1] >= \n",cnf_matrix_test[1,1])

























