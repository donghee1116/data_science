import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#data retrieve

data = pd.read_csv('../data/abalone.csv')
#print("\ndata.shape : \n", data.shape)
#print("\ndata.describe() : \n", data.describe())

#print("\n\ndata.info() : \n", data.info())

print(data.head())

label = data['Sex']
del data['Sex']


#################################################################################
#1. minmax scaling

#scaling 전 데이터
print("\ndata.head() : \n", data.head())
#print("\ndata.min() : min_values \n ", data.min())
#print("\ndata.max() : max_values \n ", data.max())

#scaler 정의
mscaler = MinMaxScaler()
print("\nMinMaxScaler() : \n", mscaler)


#scaler 적합 및 수행
mscaler.fit(data)     #fit이란, 함수를 적용시키겠다는 소리임. 그리고 transform 해야함. 실제 transform을 해야 데이터가 변환이 되는거임!!
print("\nmscaler.fit(data) : \n", mscaler.fit(data))


mMscaled_data = mscaler.transform(data)
print("\nmscaler.transform(data) : \n", mscaler)

mMscaled_data_f = pd.DataFrame(mMscaled_data, columns= data.columns)
#여기서는 팬다스에서 제공하는 표의 형태로 data를 바꿔주겠다는 소리
#data.columns 은 data에 있는 컬럼만 빼오겠다는 소리
print("\npd.DataFrame(mMscaled_data, columns= data.columns) : \n", mscaler)


#scaling 후 데이터
print("\nmMscaled_data_f.head() : \n ", mMscaled_data_f.head())
#print("\nmMscaled_data_f.min() : scaled_min_values \n ", mMscaled_data_f.min())
#print("\nmMscaled_data_f.max() : scaled_max_values \n ", mMscaled_data_f.max())


#####################################################################
### 2. standard sclaing -- z-score 사용함
print("\ndata.head() : \n", data.head())
print("\ndata.mean() :\n" , data.mean())
print("\ndata.std() :\n" , data.std())

#scaler 정의
sdscaler = StandardScaler()

#scaler 적합 및 수행
sdscaler.fit(data)   #평균 표준편차값 찾기
sdscaler_data = sdscaler.transform(data) #변환
sdscaler_pd = pd.DataFrame(sdscaler_data, columns=data.columns)

#scaling 후 데이터
print("\nsdscaler_pd.head(): \n" , sdscaler_pd.head())
print("\nsdscaler_pd.mean() :\n" , sdscaler_pd.mean())
print("\nsdscaler_pd.std() :\n" , sdscaler_pd.std())

























