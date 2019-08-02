#모두의 딥러닝. page 228
#CNN Convolution Neural Network 

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


#seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

#데이터 불러오기 
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print("X_train : \n", X_train)
print("X_train.shape : \n", X_train.shape)
print("type(X_train)", type(X_train))
print("Y_train : \n", Y_train)
print("Y_train.shape : \n",Y_train.shape)
print("Y_test : \n", Y_test)

#이미지 한개만 불러와보기. 
#plt.imshow(X_train[0], cmap ='Greys')
#plt.show()

#픽셀의 밝기 정도: 0-255 사이. 
#import sys
#for x in X_train[0]:
#    for i in x:
#        sys.stdout.write('%d\t' % i)
#    sys.stdout.write('\n')

X_train = X_train.reshape(X_train.shape[0], 28,28,1).astype('float32')/255
X_test = X_test.reshape(X_test.shape[0], 28,28,1).astype('float32')/255
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
#여기 위에 4줄은 CNN에 들어가기 위한 모델에 구조를 맞추기 위함. 

print("(Reshape) X_train.shape : \n", X_train.shape) #(60000, 28, 28, 1) 의미: 60000만장의 28*28 짜리 데이터가 있는데 그걸 한장씩 처리하겠다는 소리. 
print("(Reshape) X_test.shape : \n", X_test.shape)
print("(Reshape) Y_train.shape : \n",Y_train.shape)
print("(Reshape) Y_test.shape : \n", Y_test.shape)

################3
#여기까지 데이터를 정제했으므로, 집어 넣기만 하면 됨. 


#컨볼루션 신경망 설정
model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), input_shape = (28,28,1), activation = 'relu')) #이거의  output: 26, 26, 32
model.add(Conv2D(64, (3,3), activation = 'relu'))  #이거의  output: 24, 24, 64 
model.add(MaxPooling2D(pool_size = 2)) #반으로 짤리고, 홀수면 내림. 11이였으면 5가 되는거.   #이거의  output: 12, 12, 64
model.add(Dropout(0.25)) # 바로 위에꺼에서 자르겠다는 소리. 
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5)) #이거 하면 64개의 노드만 사용하는거임. 노드를 줄인건 아니므로 param의 갯수는 그대로임! 
model.add(Dense(10, activation = 'softmax')) #sorfmax는 분류 (한개씩 나눠줌). 그러므로 맨 마지막에 하는거임. 0-9밖에 안나옴. 데이터 자체가 0-9 이므로.
#model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#loss = 'categorical_crossentropy' : activation이 softmax이면 loss로는 categorical_crossentropy가 많이 나옴. 
 



#모델 최적화 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath=modelpath, monitor = 'val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10)
#EarlyStopping: accuracy 가 정점에 도달했을 시, 계속 할 필요가 없음. val_loss를 모니터해서, 10번 이상 좋은 값이 나오면, 중지시키겠다. 

#모델의 실행 
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = 5, batch_size= 200, verbose=0, callbacks = [early_stopping_callback, checkpointer])
#fit:  실행하는거. train을 시키는것. Y_train은 (60000, 10). 아까 reshape했으므로. 

#테스트 정확도 출력 
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

#테스트셋의 오차
y_vloss = history.history['val_loss']

#학습셋의 오차 
y_loss = history.history['loss']

#그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker = '.', c='red', label='Testset_loss')
plt.plot(x_len, y_loss, marker = '.', c='blue', label='Trainset_loss')

#그래프에 그리드를 주고 레이블을 표시 
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()









