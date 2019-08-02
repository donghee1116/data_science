# 1. 데이터 구성
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) 
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 데이터의 양을 늘림

x_test= np.array([11,12,13,14,15,16,17,18,19,20])
y_test= np.array([11,12,13,14,15,16,17,18,19,20])

x3 = np.array([101,102,103,104,105])
x4 = np.array(range(30, 50))

# 2. 모델 구성
from keras.models import Sequential 
from keras.layers import Dense
model = Sequential()

model.add(Dense(200, input_dim=1, activation='relu')) 
model.add(Dense(22))
model.add(Dense(13))
model.add(Dense(62))
model.add(Dense(1))

# 3. 훈련
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=30, batch_size=2) 

# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", acc)
print("loss : ", loss) #loss 의 값이 나올꺼 


y_predict = model.predict(x_test)
#y_predict = model.predict(x4)  # x_test값을 넣었을 때의 모델을 예측하라 / acc 외의 평가지표를 추가한 것
print(y_predict) # 11.0795, 12.024, 12.969.... 잘 구축된 모델임


#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))










