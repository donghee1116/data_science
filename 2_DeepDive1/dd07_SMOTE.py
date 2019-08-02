
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from plotnine import *
import pandas as pd


'''
##2. SMOTE 기본 code Frame
smote = SMOTE(k_neighbors=5, random_state=2019)

smoted_data, smoted_label = smote.fit_resample(X_train, Y_train)  
smoted_data = pd.DataFrame(smoted_data, columns=data.columns)
smoted_label = pd.DataFrame({'Sex' : smoted_label})

print("원본데이터의 클래스 비율 \n {}".format(pd.get_dummies(Y_train).sum()))
print("\nsmoted_data 클래스 비율 \n{}".format(pd.get_dummies(smoted_label).sum()))
'''


##################################
#titanic에 적용

#data load
data = pd.read_csv('../data/titanic_proc.csv')
print(data.info())
print(data.head(5))
print("data.columns \n", data.columns)

data['Survived'] = data['Survived'].astype(str)
label = data['Survived']
del data['Survived']




#scaling
sdscaler = StandardScaler()
sdscaler.fit(data)   #평균, 표준편차값 찾기   #fit이란, 함수를 적용시키겠다는 소리임. 그리고 transform 해야함. 실제 transform을 해야 데이터가 변환이 되는거임!!
sdscaler_data = sdscaler.transform(data) #변환
sdscaler_pd = pd.DataFrame(sdscaler_data, columns=data.columns)

print("\nsdscaler_pd : \n", sdscaler_pd)

#성능 비교를 위한 test set 설정
X_train, X_test, Y_train, Y_test = train_test_split(sdscaler_pd, label, test_size= 0.1, shuffle=True, random_state=5)
#shuffle=True 는 섞으란 소리
print("\n\nX_train[0:10]\n",X_train[0:10])
print("\n\nY_train[0:10]\n",Y_train[0:10])
#train, test set 분리 후 svm 분류모델을 통해 성능 비교
smote = SMOTE(k_neighbors=3, random_state=2019)
smoted_data, smoted_label = smote.fit_resample(X_train, Y_train) #왜 여기서는 model.fit 이 아니라 model.fit_resample이지??
smoted_data = pd.DataFrame(smoted_data, columns = data.columns)
smoted_label = pd.DataFrame({'Survived' : smoted_label}) # {} 가 의미하는것?

print("원본데이터의 클래스 비율 \n {}".format(pd.get_dummies(Y_train).sum()))
print("\nsmoted_label 클래스 비율 \n{}".format(pd.get_dummies(smoted_label).sum()))

#성능 비교용 함수 카피해오기
def train_and_test(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train,Y_train)
    prediction = model.predict(X_test)
    accuracy = round(accuracy_score(Y_test, prediction)*100, 2)
    print("Accuracy: \n", accuracy, "%")

print("original_data :\n ", train_and_test(SVC(), X_train, Y_train, X_test, Y_test))

print("smoted_data : \n", train_and_test(SVC(), smoted_data, smoted_label, X_test, Y_test))


#시각화 확인
def ggplot_point (X_train, Y_train, x, y):
    data = pd.concat([X_train, Y_train], axis = 1)
    plot = (ggplot(data)+aes(x=x, y=y, fill = 'factor(Survived)') + geom_point())
    print(plot)

print("X_train.columns[2]: \n", X_train.columns[2])   #Fare
print("X_train.columns[2]: \n", X_train.columns[3])     #Family
print("smoted_data.columns[2]: \n", smoted_data.columns[2])
print("smoted_data.columns[2]: \n", smoted_data.columns[3])


ggplot_point(X_train, Y_train, X_train.columns[2], X_train.columns[3])
ggplot_point(smoted_data, smoted_label, smoted_data.columns[2], smoted_data.columns[3])

#plot 창이 따로 안뜨게 할수는 없는지..?









