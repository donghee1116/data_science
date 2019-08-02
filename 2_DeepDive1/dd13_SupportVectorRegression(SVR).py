
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from data import bostonData
import matplotlib.pyplot as plt
import numpy as np
svm_regr = SVR()


#학습
svm_regr.fit(bostonData.x_train['RM'].values.reshape((-1,1)),bostonData.y_train)

#예측
y_pred = svm_regr.predict(bostonData.x_test["RM"].values.reshape((-1,1)))

#평가
print("SVM 회기: \n", format(r2_score(bostonData.y_test, y_pred)))

#plot 그림
plt.scatter(bostonData.x_test['RM'], bostonData.y_test, s= 10, c='black')
line_x = np.linspace(np.min(bostonData.x_test['RM']),np.max(bostonData.x_test['RM']),100)
line_y = svm_regr.predict(line_x.reshape(-1,1))

#plt.plot(bostonData.x_test['RM'], y_pred, c = 'red')
plt.plot(line_x, line_y, c='red')

plt.legend(['SVM line', 'x_test'], loc = 'upper left')
plt.show()

#전체 변수 입력
svm_regr = SVR()
svm_regr.fit(bostonData.x_train, bostonData.y_train)
y_pred = svm_regr.predict(bostonData.x_test)

print("SVM 전체 회기: \n", format(r2_score(bostonData.y_test, y_pred)))





















