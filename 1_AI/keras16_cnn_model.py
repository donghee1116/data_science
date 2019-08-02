from keras.models import Sequential
from keras.layers import Conv2D, Flatten


#filter_size = 32 #변수
#kernel_size = (3,3) #변수 
model = Sequential()

#model.add(Conv2D(filter_size, kernel_size, padding = 'same', input_shape = (28,28,1))) #padding = 'valid' #28 by 28 짜리 1장씩
model.add(Conv2D(32, (2,2), padding = 'same', input_shape = (7,7,1))) # 윗 라인과 결국 같은것. 
model.add(Conv2D(16, (2,2)))

from keras.layers import MaxPooling2D
pool_size = (2,2)
model.add(MaxPooling2D(pool_size=2)) #수치가 절반이 되는것 같음. 수영장에서 최고치를 찾아라 -> 특이점


model.add(Flatten()) #평평하게 일자로 쭉 나열하는것. 





model.summary()










