# X라는 변수를 x_train, x_val, x_test 로 내가 나누는게 아니라, 머신이 알아서 split하도록 하는게 목표


#x-val 이라는 data를 추가로 넣어줌. model.fit에서 validation_data 추가해줌. 
# 1. 데이터 구성
import numpy as np

xxx= np.array(range(100))
yyy= np.array(range(100))

#list 형태로 자름
x_train = xxx[:60]
y_train = yyy[:60]
x_val = xxx[60:80]
y_val = yyy[60:80]
x_test = xxx[80:]
y_test = yyy[80:]

#일일히 숫자를 계산하고 있을 순 없자나. 그럼 어떡해? 이미 만들어져 있는 거를 가져온다. -> keras_10

print("x_train.shape", x_train.shape)
print("x_val.shape", x_val.shape)
print("x_test.shape", x_test.shape)


#x_train = np.array([1,2,3,42,5,6,7,8,59,10]) # 행 무시 열 우선. 그럼 20% 문제가 되는 데이터를 지우거나 수정. 밑에 train, test꺼까지 다. 
#x_train = np.array([1,2,3,5,6,7,8,10])
#y_train = np.array([1,2,4,5,6,7,9,10]) # 데이터의 양을 늘림
#x_val = np.array([101,102,103,104,105])
#y_val = np.array([101,102,103,104,105])

#x_test= np.array([1001,1002,1003,1004,1006,1008,1009,1010])
#y_test= np.array([1001,1002,1004,1005,1006,1207,1608,1310])

#x3 = np.array([101,102,103,104,105])
#x4 = np.array(range(30, 50))

# 2. 모델 구성
from keras.models import Sequential 
from keras.layers import Dense
model = Sequential()

model.add(Dense(7, input_dim=1, activation='linear')) 
#input dimension = 1 , 맨 마지막 output도 1 이어야. 
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

#model.fit(x_train, y_train, epochs=50, batch_size=1, validation_data = (x_test, y_test)) 
            #이거의 문제점: test data와 validation data가 같음. 즉, overfit이 남. 
            #x-train: Machine, x-validation: Machine, x-test: human 
            #근데 사람이 테스트 할 똑같은 데이터를 기계가 validation할때 써버림. 모의고사와 본고사가 같은 시험지인 격. 

model.fit(x_train, y_train, epochs=73, batch_size=2, validation_data = (x_val, y_val)) 
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

















