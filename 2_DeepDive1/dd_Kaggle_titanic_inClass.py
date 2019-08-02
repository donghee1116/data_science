#hyper parameter 빼고 한 light 버전

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#acquire data
train = pd.read_csv('../data/titanicK_train.csv')
print(train)
#print(train_df.columns)
print(train.columns.values)



test = pd.read_csv('../data/titanicK_test.csv')
print(test)

print("train.shape",train.shape)
print("train.info()", train.info())
print("test.shape",test.shape)
print("test.info()", test.info())

def pie_chart(feature):
    feature_ratio = train[feature].value_counts(sort = False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived']== 0][feature].value_counts()
    print("feature_size\n", feature_size)
    print("feature_index\n", feature_index)
    print("survived count\n", survived)
    print("dead count\n", dead)
    plt.plot(aspect = 'auto')
    plt.pie(feature_ratio, labels = feature_index, autopct = '%1.1f%%')
    plt.title(feature+'\'s ratio in total')
    plt.show()

    for i, index in enumerate(feature_index):
        plt.subplot(1, feature_size+1, i+1, aspect = 'equal')
        plt.pie([survived[index], dead[index]], labels = ['Survived', 'Dead'],autopct ='%1.1f%%')
        plt.title(str(index)+'\'s ratio')

    plt.show()


def bar_chart(feature):
    survived = train[train['Survived'] ==1][feature].value_counts()
    dead = train[train['Survived'] ==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])

    print("survived",survived)
    print("dead",dead)
    df.index = ['Survived','Dead']
    df.plot(kind = 'bar', stacked = True)
    plt.show()

if __name__ == '__main__':
    pie_chart('Sex')
    pie_chart('Pclass')
    pie_chart('Embarked')
    bar_chart('SibSp')
    bar_chart('Parch')
    bar_chart('Embarked')


# 데이터 전처리 및 특성 추출
# sex feature : male과 female => string data로 변경

#Sex, Embarked, Age, SibSp, Parch, Fare, Pclass

dataset_ = [train,test]

#sex feature
for dataset in dataset_:
    dataset['Sex'] = dataset['Sex'].astype(str) #string 변수로 변형

#embarked feature
print("train.isnull().sum()", train.isnull().sum())
train['Embarked'].value_counts(dropna=False)

for dataset in dataset_:
    dataset['Embarked'] = dataset['Embarked'].fillna('S') #null값 채우기
    dataset['Embarked'] = dataset['Embarked'].astype(str)
print("train.isnull().sum()", train.isnull().sum())

#age feature
for dataset in dataset_:
    dataset['Age'].fillna(dataset['Age'].mean(), inplace = True)
    dataset['Age'] = dataset['Age'].astype(int)
    train['AgeBand'] = pd.cut(train['Age'],5)
print(train[['AgeBand','Survived']].groupby(['AgeBand'], as_index = False).mean())  #survived ratio about AgeBand

for dataset in dataset_:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].map({0: 'Child', 1:'Young', 2:'Middle', 3:'Prime',4:'Old'}).astype(str)

#sibsp & Parch Feature
for dataset in dataset_:
    dataset['Family'] = dataset["Parch"]+dataset["SibSp"]
    dataset['Family'] = dataset['Family'].astype(int)


#가공된 데이터셋 확인
features_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(features_drop, axis = 1)
test = test.drop(features_drop, axis = 1)
train = train.drop(['PassengerId', 'AgeBand'], axis = 1)

print(train.head())
print(test.head())
print("train.isnull().sum()", train.isnull().sum())
print("test.isnull().sum()", test.isnull().sum())

#one-hot-encoding for categorical variabels
train = pd.get_dummies(train)
test = pd.get_dummies(test)

train_label = train['Survived']
train_data = train.drop('Survived', axis = 1)
test_data = test.drop("PassengerId",axis = 1).copy()
test_data.fillna(test_data.mean())  #test 셋 null값 채워넣기
print(train_data.columns)
print(test_data.columns)

test_data_co = test_data.fillna(test_data.mean())


#가공된 데이터셋 나누기

def train_and_test(model, train_data, train_label):
    from sklearn.metrics import accuracy_score

    X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_label, test_size = 0.2, shuffle = True, random_state = 5)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    accuracy = round(accuracy_score(Y_test, prediction)*100,2)
    print("Accuracy : ", accuracy, "%")
    return prediction

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#모델넣기
log_pred = train_and_test(LogisticRegression(),train_data, train_label)

#SVM
svm_pred = train_and_test(SVC(), train_data, train_label)

#random fores
rf_pred = train_and_test(RandomForestClassifier(n_estimators=100),train_data,train_label)



submission = pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":svm_pred})
submission.to_csv('submission_rf.csv', index=False)



















