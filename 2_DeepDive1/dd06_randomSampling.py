#p 28 Random Sampling 부터

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#데이터 불러오기
data = pd.read_csv("../data/abalone.csv")
print("data.head() : \n",data.head())
#print("data.columns: \n", data.columns)
label = data['Sex']
del data['Sex']
print("data.describe() : \n",data.describe())

#scaling
sdscaler = StandardScaler()
sdscaler.fit(data)  # 평균 표준편차값 찾기
# #fit이란, 함수를 적용시키겠다는 소리임. 그리고 transform 해야함. 실제 transform을 해야 데이터가 변환이 되는거임!!
sdscaler_data = sdscaler.transform(data)
sdscaler_pd = pd.DataFrame(sdscaler_data, columns=data.columns)

#성능 비교를 위한 test set 설정
X_train, X_test, Y_train, Y_test = train_test_split(sdscaler_pd, label, test_size= 0.1, shuffle=True, random_state=5)


#Random Sampling
ros = RandomOverSampler(random_state=2019)
rus = RandomUnderSampler(random_state=2019)

oversampled_data, oversampled_label = ros.fit_resample(X_train, Y_train)
undersampled_data, undersampled_label = rus.fit_resample(X_train, Y_train)
oversampled_data = pd.DataFrame(oversampled_data, columns=data.columns)
#print("data.columns: \n", data.columns)
undersampled_data = pd.DataFrame(undersampled_data, columns = data.columns)

print("pd.get_dummies(Y_train) : \n", pd.get_dummies(Y_train))
print("pd.get_dummies(Y_train).sum() : \n", pd.get_dummies(Y_train).sum())

print("원본데이터의 클래스 비율 \n ", format(pd.get_dummies(Y_train).sum()))
#pd.get_dummies(Y_train) ??????
print("\noversampled_data 클래스 비율 \n".format(pd.get_dummies(oversampled_label).sum()))
print("\nundersampled_data 클래스 비율 \n".format(pd.get_dummies(undersampled_data).sum()))


#성능 비교
def train_and_test(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train,Y_train)
    prediction = model.predict(X_test)
    accuracy = round(accuracy_score(Y_test, prediction)*100, 2)
    print("Accuracy: \n", accuracy, "%")

print("original_data \n") ; train_and_test(SVC(), X_train, Y_train, X_test, Y_test)

print("oversampled_data \n") ; train_and_test(SVC(), oversampled_data, oversampled_label, X_test, Y_test)


print("undersampled_data  \n") ; train_and_test(SVC(), undersampled_data, undersampled_label, X_test, Y_test)






















