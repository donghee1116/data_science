
from sklearn.tree import DecisionTreeRegressor
from data import bostonData
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

import graphviz
from sklearn.tree import export_graphviz

from mglearn.plots import plot_animal_tree

plot_animal_tree()
plt.show()



#non-overfitting
dt_regr = DecisionTreeRegressor(max_depth=5, random_state=2019)

#2. 모델 학습
dt_regr.fit(bostonData.x_train['RM'].values.reshape(-1,  1), bostonData.y_train)

#예측
y_pred = dt_regr.predict(bostonData.x_test['RM'].values.reshape(-1,1))

print("단순 결정 트리 회기: ", format(r2_score(bostonData.y_test, y_pred)))

#10개만
line_x = np.linspace(np.min(bostonData.x_test['RM']), np.max(bostonData.x_test['RM']),10)
line_y = dt_regr.predict(line_x.reshape(-1,1))

#10개만 선별해서 그림
plt.scatter(bostonData.x_test['RM'].values.reshape(-1,1), bostonData.y_test, s = 10, c='black')
plt.plot(line_x, line_y, c='red')
plt.legend(['DT Regression line','x_test'], loc = 'upper left')
plt.show()

#13개 변수 사용 test
#학습
dt_regr.fit(bostonData.x_train, bostonData.y_train)
y_pred_all = dt_regr.predict(bostonData.x_test)
print("단순 결정 트리 (all 변수) 회 : \n", format(r2_score(bostonData.y_test, y_pred_all)))




'''
import matplotlib.pyplot as plt
from mglearn.plots import plot_animal_tree

plot_animal_tree()
plt.show()
'''


















