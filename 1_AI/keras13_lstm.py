#RNN: Recurrent Neural Network 
#LSTM: Long Short-Term Memory network - RNN의 한가지 기법 중 하나. 보통 RNN은 LSTM으로 함. 
#맨 뒤에 작업량 (실행갯수)를 써줘야하는 특징이 있음
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# RNN의 핵심은 다음 숫자 맞추기!! 
# DNN은 열만 맞추면 됐음. RNN은 몇 행 몇 열 + 몇개씩 들어가느냐 .
# 현재의 데이터는 과거의 데이터에 계속 영향을 받음 

from numpy import array 
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

#1. 데이터 
X = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]) #4행 3열임. 
y = array ([4,5,6,7]) # python은 1행 4열로 인식하게 됨.    

print("X.shape : ", X.shape) #(4,3) 나옴. 4행 3열 이란 소리. 
print("y.shape : ", y.shape)


#reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))  #RNN이 요구하는 규격을 맞추기 위해 reshape! 
                                #X.shape 는 (4,3) 4행 3열 인거자나. 그러므로 X.shape[1]은 4인것.  

print("X.shape : ", X.shape)  # -> (4,3) 4행 3열에 몇개씩 작업하느냐 (1 - batch size와 비슷함): (4,3,1) 
print("y.shape : ", y.shape)

#2.모델구성

model = Sequential()
model.add(LSTM(50, activation = 'relu', input_shape = (3,1))) #LSTM은 순차적인걸 계산하는거. #50은 첫번째 아웃풋. 인풋은 (4,3,1)에서 행은 필요 없으므로 (3,1) 이라 적음. 
#model.add(LSTM(50)) #LSTM을 연속해서 사용하면 오류남. 뭔가를 더 이어줘야하는데 구글링 하라 함. 
#model.add(LSTM(50, return_sequence = True)) #return_sequence = True 넣으면 괜찮아짐. 
model.add(Dense(30)) #맨 처음만 LSTM을 하고 하단은 Dense로 연결함. LSTM으로 해도 상관은 없으나 시간이 오래 걸림. 
model.add(Dense(72))
model.add(Dense(13))
model.add(Dense(5))
model.add(Dense(1)) #y값 원하는게 1개 이므로, dense = 1
model.compile(optimizer = 'adam', loss = 'mse')

#3.실행
model.fit(X, y, epochs = 300, verbose = 2)
#demonstrate prediction 
x_input = array ([6, 7, 8])   #70, 80, 90 => ? 

print("x_input.shape : ", x_input.shape)

#x_input = array ([70,80,90])  #70, 80, 90 => 38.37 나옴. 왜? 머신이 공부한 데이터는 1단위씩 늘어난것. 데이터를 바꾸지 않는 한, 제대로 된 예측 못해줌. 
x_input = x_input.reshape((1, 3, 1))  # (1,3) 1 행 3열짜리에, 마지막 1은 batch_size. 
print("x_input.shape_2 : ", x_input.shape)
yhat = model.predict(x_input, verbose = 0) # 6,7,8을 넣었을 때 뭘 예측하느냐
print(yhat)


#여기도 evaluation하는 함수들이 있음. 하지만 시간 관계상 나가진 않음. 







