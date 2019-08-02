import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
# from keras.utils import np_utils
# a = np.array([11,12,13,14,15,16,17,18,19,20])
a = np.array(range(11,21))
# a = np.array(range(100))

print("a : " , a)
window_size = 5
def split_5(seq, window_size):  # 데이터를 5개씩 자르기용.    # 입력이 5이고 5개씩 자르기
    aaa = []
    for i in range(len(a)-window_size +1):                 # 열
        subset = a[i:(i+window_size)]       # 0~5
        aaa.append([item for item in subset])
        # print(aaa)
    print(type(aaa))    
    return np.array(aaa)

dataset = split_5(a, window_size)     # 5씩 잘랏으니 (5, 6)가 된다. // window_size+1 만큼씩 잘라진다.
print("===========================")
print("dataset: \n", dataset)
print("dataset.shape : \n", dataset.shape)    # (6, 5)

#입력과 출력을 분리시키기  5개와 1개로
print("type(dataset): \n",type(dataset))
x_train = dataset[:,0:4]  #행은 다 원하고 열은 [0]~[3]까지를 원한다. 
y_train = dataset[:,4]

print("x_train : \n", x_train)
print("y_train : \n", y_train)


x_train = np.reshape(x_train, (len(a)-window_size+1, 4, 1))  #(4,4,1) 이거의 목적??? 1은 1회 작업량임. 

x_test = np.array([[[21],[22],[23],[24]], [[22],[23],[24],[25]], 
                  [[23],[24],[25],[26]], [[24],[25],[26],[27]]])
y_test = np.array([25, 26, 27, 28])  #여긴 왜 [] 안씀? 

print("updated x_train \n", x_train)

print("x_test \n", x_test)
print("y_test \n", y_test)
print("x_train.shape \n", x_train.shape)    # (6, 4, 1)
print("y_train.shape \n", y_train.shape)    # (6, )
print("x_test.shape \n", x_test.shape)     # (4, 4, 1)
print("y_test.shape \n", y_test.shape)     # (4, )


'''
# 모델 구성하기
model = Sequential()

model.add(LSTM(32, input_shape=(4,1), return_sequences=True))  #LSTM을 사용할 때엔 많은 데이터가 있어야 좋음 
                                                                #층이 많다고 무조건 좋은게 아님. 

#model.add(LSTM(10, return_sequences=True))
#model.add(LSTM(10, return_sequences=True))
#model.add(LSTM(10, return_sequences=True))
#model.add(LSTM(10, return_sequences=True))
#model.add(LSTM(10, return_sequences=True))
#model.add(LSTM(10, return_sequences=True))
#model.add(LSTM(10, return_sequences=True))

model.add(LSTM(10))



# model.add(Dropout(0.2))
model.add(Dense(5, activation='relu'))
model.add(Dense(52, activation='relu'))
model.add(Dense(54, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(34, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))


model.add(Dense(1))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)

loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
y_predict2 = model.predict(x_train)

print('loss : ', loss)
print('acc : ', acc)
print('y_predict(x_test) : \n', y_predict)
print('y_predict2(x_train) : \n', y_predict2)

print(a)

'''

