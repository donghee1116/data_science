import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#원본 데이터 값
ohe = OneHotEncoder(categorical_features=[0]) #Index 0 data one-hot coding
#print("ohe: \n", ohe)
ohe = ohe.fit_transform(x).toarray() #array 로 바꿔줘
print("label 후 원핫인코딩 값 : \n", ohe)


#pandas
x_df = pd.get_dummies(df[['color', 'size', 'price', 'type']])
print("Dummy 후 변환값: \n", x_df)





