from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

def train_test_split_():
    iris = load_iris()
    x=iris.data
    y=iris.target

    #데이터와 타겟 레이블을 훈련 세트와 테스트 세트로 나눔
    x_train,x_test, y_train, y_test = train_test_split(x,y,random_state=0)

    #모델 학습
    logreg = LogisticRegression().fit(x_train,y_train)
    #테스트 세트 평가
    print("테스트 세트 점수: {:.2f}".format(logreg.score(x_test,y_test)))




#=====================================================

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def k_fold():
    iris = load_iris()
    kf_data = iris.data   #fit 될 데이터
    kf_label = iris.target     #predict할 목표변수
    kf_columns = iris.feature_names

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2019)  #rf: 데이터를 fit할 객체
    scores = cross_val_score(rf, kf_data, kf_label, cv = 10)   #cv 만큼의 fold를 만듬

    print("scores",scores)
    print('rf k-fold CV score:{:.2f}%'.format(scores.mean()))

from IPython.display import display
import pandas as pd

def k_fold_validate():
    from sklearn.model_selection import cross_validate
    iris = load_iris()
    kf_data = iris.data
    kf_label = iris.target
    kf_columns = iris.feature_names

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2019)  #rf: 데이터를 fit할 객체
    scores = cross_validate(rf, kf_data, kf_label, cv = 10, return_train_score=True)   #cv 만큼의 fold를 만듬

    print("<< score >>")
    display(scores)
    res_df = pd.DataFrame(scores)
    print("<< res_df >>")
    display(res_df)
    print("평균 시간과 점수\n", res_df.mean())

#함수 호출
if __name__ =='__main__':
    train_test_split_()
    k_fold()
    k_fold_validate()
































