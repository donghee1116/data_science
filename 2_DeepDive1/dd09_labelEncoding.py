import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.DataFrame([["yellow", 'M','23','a'], ["red", "L",'26','c'], ["blue",'XL','20','c']])
df.columns = ['color', 'size', 'price','type']
print(df)

#데이터셋 필요한 부분 숫자 변경

x = df[['color', 'size','price','type']].values #데이터 프레임에서 numpy.narray로 변화
print(x)


### 1. Label Encoding

shop_le = LabelEncoder() #string을 int라벨로 변화

x[:,0] = shop_le.fit_transform(x[:,0])    # 뭐하는거?
x[:,1] = shop_le.fit_transform(x[:,1])
x[:,2] = x[:,2].astype(dtype = float) #string이었던 price 를 float 로 바꿔줌
x[:,3] = shop_le.fit_transform(x[:,3])

print("label encoder 변환값: \n", x)

####################################################
print("\n\n==============================\n\n")
### 2. One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder

#원본 데이터 값
ohe = OneHotEncoder(categorical_features=[0]) #Index 0 data one-hot coding #뭐 하는거임 ?
print("ohe: \n", ohe)
ohe = ohe.fit_transform(x).toarray() #array 로 바꿔줘
print("label 후 원핫인코딩 값 : \n", ohe)


#pandas
x_df = pd.get_dummies(df[['color', 'size', 'price', 'type']])
print("Dummy 후 변환값: \n", x_df)




























