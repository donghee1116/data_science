######bostonData 불러오기 (그냥 bostonData.py 카피페이스트 함)
import pandas as pd

# data retrieve from sklearn web site
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


boston = load_boston()  #너누 길어서 boston으로 할당함
print("boston.DESCR: \n", boston.DESCR)   #주택 가격에 대한 설명
data = boston.data
label = boston.target
print("label: \n", label)
print("data: \n", data)

columns = boston.feature_names#columns name 알고 싶을 때
print("columns : \n", columns)


#데이터 프레임 변화
data = pd.DataFrame(data, columns = columns)
print("data.head(): \n", data.head())
print("data.shape\n",data.shape)
print("data.describe(): \n", data.describe())
print("data.info: \n", data.info) # 자료형

#데이터 나누기

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=2019)

#모델 불러오기
from sklearn.linear_model import LinearRegression
sim_lr = LinearRegression()
sim_lr.fit(x_train['RM'].values.reshape((-1,1)), y_train)
y_pred = sim_lr.predict(x_test['RM'].values.reshape((-1,1)))

#####     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

sim_lr = LinearRegression()
sim_lr.fit(boston.x_train)










