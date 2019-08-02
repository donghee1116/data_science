# ensemble - 변수 여러개를 합쳐서 하나의 모델을 만듦. 


import numpy as np


xxx1 = np.array([range(100), range(311, 411)]) #input. 뒤에 x_train, val, test로 나눴어도 어쨌든 변수는 1개임. 
xxx2 = np.array([range(100), range(511, 611)])
yyy1 = np.array([range(201, 301), range(211, 111, -1)]) #output.
yyy2 = np.array([range(501, 601), range(111, 11, -1)])
#2행 10열. 
#range(111, 11, -1) 이라 치면 기울기가 음수 됨. 
print(xxx1.shape)
#xxx = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]])
#yyy = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]])
#print("First xxx: ", xxx)
## 10행 2열로 바꿈. 1열이 0~9, 2열이 11~20 이 나옴 
xxx1 = np.transpose(xxx1)
xxx2 = np.transpose(xxx2)
yyy1 = np.transpose(yyy1)
yyy2 = np.transpose(yyy2)
#print("Second xxx: ", xxx)
print(xxx1.shape)

#print(xxx.shape)
#print(yyy.shape)



# model 1
from sklearn.model_selection import train_test_split
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(xxx1, yyy1 ,test_size = .40)
#default: 75%:25%
# test_size=0.20

#한번 더 나눔. 
x_val_1, x_test_1, y_val_1, y_test_1 = train_test_split(x_test_1,y_test_1,test_size = .5, random_state = 66)


#model 2
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(xxx2, yyy2 ,test_size = .40)
#한번 더 나눔. 
x_val_2, x_test_2, y_val_2, y_test_2 = train_test_split(x_test_2,y_test_2,test_size = .5, random_state = 66)


print('x_train_1: ', x_train_1)
print("x_train_1.shape: ", x_train_1.shape)

# 2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate
from keras.layers.merge import concatenate
#model_1 = Sequential() -> sequential을 못쓰는 이유: 인풋이 두개이기 때문에. 씨퀀셜은 하나의 인풋일때만 가능한거임. 
# 그래서 함수형 모델로 바꿈. Input을 사용하는 함수형 모델.

#model 1
input1 = Input(shape=(2,)) # input layer: 2개 입력.  ----- 여기까지는 아직 xxx1, xxx2 데이터 안씀. 일단 열만 맞춰놓은거임. 
dense1 = Dense(100, activation = 'relu')(input1) #dense: 출력 100개


#model 2
input2 = Input(shape=(2,))
dense2 = Dense(50, activation = 'relu')(input2)
dense2 = Dense(50, activation = 'relu')(dense2)

#merge
#merge = Concatenate[dense1, dense2]
merge1 = concatenate([dense1,dense2]) 

output_11 = Dense(10)(merge1)
output_12 = Dense(5)(output_11)
merge2 = Dense(3)(output_12) #model 3
###함수형이지만 sequential 모델과 같음


output1=Dense(30)(merge2) #model 4
output1=Dense(2)(output1)

output2=Dense(70)(merge2) #model 5
output2=Dense(2)(output2)

#모델 정의 
model = Model (inputs = [input1, input2], outputs = [output1, output2])

#model.summary()


#3. 훈련

model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse'])
model.fit([x_train_1, x_train_2], [y_train_1, y_train_2], 
            epochs=50, batch_size=1, 
            validation_data=([x_val_1, x_val_2], [y_val_1, y_val_2])) #변수가 2개 이상이면 list로 넣으면 됨. 

#4. 평가 예측

acc = model.evaluate([x_test_1, x_test_2], [y_test_1, y_test_2], batch_size=1)
#print("acc: ", acc)

y_predict_1, y_predict_2 = model.predict([x_test_1, x_test_2]) #왜 리스트로 넣어야함? 하나의 입력값을 넣어야 하니까 두개의 입력값을 하나의 리스트로 넣는다는 소리. 
print("y_predict_1 & y_predict_2 : \n", y_predict_1, y_predict_2)


#RMSE - 낮을수록 좋음 
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) #y_test, y_predict는 이 함수 안에서만 쓰는거이므로 상관 없음. 
RMSE1 = RMSE(y_test_1, y_predict_1) 
RMSE2 = RMSE(y_test_2, y_predict_2) 
print("RMSE1 : \n", RMSE1)
print("RMSE2 : \n", RMSE2)
print("RMSE AVG: \n", (RMSE1 + RMSE2)/2)


#R2 (결정계수) 구하기 -- 1에 가까울수록 좋음. RMSE가 낮을수록 R2가 1에 가깝게 나옴
from sklearn.metrics import r2_score
r2_y_predict_1 = r2_score(y_test_1, y_predict_1)
r2_y_predict_2 = r2_score(y_test_2, y_predict_2)
print("r2_y_predict_1 : \n", r2_y_predict_1) 
print("r2_y_predict_2 : \n", r2_y_predict_2) 

print("r2_y_predict_AVG : \n", (r2_y_predict_1 + r2_y_predict_2)/2) 




















