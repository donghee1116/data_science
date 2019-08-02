# MLP = Multilayer Perceptron (다층 인공인지체)


import numpy as np
from sklearn.model_selection import train_test_split

xxx = np.array([range(100), range(311, 411), range(511, 611)]) #input. 뒤에 x_train, val, test로 나눴어도 어쨌든 변수는 1개임. 
yyy = np.array([range(501, 601), range(111, 11, -1), range(311, 211, -1)]) #output.
#2행 10열. 
#range(111, 11, -1) 이라 치면 기울기가 음수 됨. 

#xxx = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]])
#yyy = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]])
#print("First xxx: ", xxx)
## 10행 2열로 바꿈. 1열이 0~9, 2열이 11~20 이 나옴 
xxx = np.transpose(xxx)
yyy = np.transpose(yyy)
#print("Second xxx: ", xxx)

#print(xxx.shape)
#print(yyy.shape)





x_train, x_test, y_train, y_test = train_test_split(xxx, yyy ,test_size = .40)
#default: 75%:25%
# test_size=0.20

#한번 더 나눔. 
x_val, x_test, y_val, y_test = train_test_split(x_test,y_test,test_size = .5, random_state = 66)


# 2. 모델 구성
from keras.models import Sequential 
from keras.layers import Dense
model = Sequential()

model.add(Dense(7, input_dim=3, activation='relu')) 
#2차원은 input dimension = 2 , 맨 마지막 output도 2 이어야. 
#activation 밑에는 안씀. 그래도 돌아간다는 말은, default가 있다는 소리. 현 상황에선 linear보단 relu가 더 좋음. 
model.add(Dense(21))
model.add(Dense(34))
model.add(Dense(91))
model.add(Dense(111))
model.add(Dense(211))
model.add(Dense(158))
model.add(Dense(139))
model.add(Dense(211))
model.add(Dense(98))
model.add(Dense(119))
model.add(Dense(78))
model.add(Dense(58))
model.add(Dense(37))
model.add(Dense(73))
model.add(Dense(93))
model.add(Dense(77))
model.add(Dense(53))
model.add(Dense(37))
model.add(Dense(3))  # 이게 맨 마지막 output. 2차원이니까 출력이 2여야. #꼭 그럴 필요는 없다. 내가 어떻게 모델링 하냐에 따라 달라진다.   
#노드가 많아도 overfitting 생길 수도 있음. 

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

#model.fit(x_train, y_train, epochs=50, batch_size=1, validation_data = (x_test, y_test)) 
            #이거의 문제점: test data와 validation data가 같음. 즉, overfit이 남. 
            #x-train: Machine, x-validation: Machine, x-test: human 
            #근데 사람이 테스트 할 똑같은 데이터를 기계가 validation할때 써버림. 모의고사와 본고사가 같은 시험지인 격. 

model.fit(x_train, y_train, epochs=30, batch_size=2, validation_data = (x_val, y_val)) 
#model.fit(x_train, y_train, epochs=73, batch_size=2, validation_data = (x_val, y_val)) 
#batch_size는 몇개씩 묶어서 test할지. model.fit의 batch size를 바꿔야 하고, loss, acc에서의 batch size는 신경 안써도 됨. 
#epoch는 총 작업량. 머신한테 몇번을 작업 시킬 것인가. 


# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=2) 
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


model.summary()



















