# 1. 데이터 구성
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) 
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 데이터의 양을 늘림

x_test= np.array([1001,1002,1003,1004,1005,1006,1007,1008,1009,1010])
y_test= np.array([1001,1002,1003,1004,1005,1006,1007,1008,1009,1010])

#x3 = np.array([101,102,103,104,105])
#x4 = np.array(range(30, 50))

# 2. 모델 구성
from keras.models import Sequential 
from keras.layers import Dense
model = Sequential()

model.add(Dense(200, input_dim=1, activation='linear')) 
#input dimension = 1 , 맨 마지막 output도 1 이어야. 
#activation 밑에는 안씀. 그래도 돌아간다는 말은, default가 있다는 소리. 현 상황에선 linear보단 relu가 더 좋음. 
model.add(Dense(2))
model.add(Dense(142))
model.add(Dense(68))
model.add(Dense(1))  # 이게 맨 마지막 output. 그래서 이건 꼭 1이어야. 

#loss: 손실률 (mse, mae가 있음)
#activation: linear
# 7가지를 변경할 수 있음. - node, dense, activation, optimizer, loss, metric, epochs, batch_size)

# 3. 훈련
#model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#metrics=['accuracy'] 덕분에 epoch가 어떻게 발전되는지 보이는거임. 

#loss를 mae, mse 둘다 가능하지만, 둘이 결과값이 당연히 다름. 그래서 그때그때 모델링 새로 해줘야.
# metrics는 보여지는 부분일 뿐.  
#optimizer는 현재까진 adam이 좋음. (rmsprop, sgd 같은게 있음. )
model.fit(x_train, y_train, epochs=50, batch_size=3) 
#batch_size는 몇개씩 묶어서 test할지. model.fit의 batch size를 바꿔야 하고, loss, acc에서의 batch size는 신경 안써도 됨. 
#epoch는 총 작업량. 머신한테 몇번을 작업 시킬 것인가. 


# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1) 
print("acc : ", acc)
print("loss : ", loss) #loss 의 값이 나올꺼 


y_predict = model.predict(x_test)
#y_predict = model.predict(x4)  # x_test값을 넣었을 때의 모델을 예측하라 / acc 외의 평가지표를 추가한 것
print(y_predict) # 11.0795, 12.024, 12.969.... 잘 구축된 모델임


#RMSE - 낮을수록 좋음 
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))


#R2 (결정계수) 구하기 -- 1에 가까울수록 좋음. RMSE가 낮을수록 R2가 1에 가깝게 나옴
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test,y_predict)
print("R2 : ", r2_y_predict) 


#keras는 deep learning. sklearn is machine learning. 









