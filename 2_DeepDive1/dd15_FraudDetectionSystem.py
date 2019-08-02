
from data import creditcard
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC

#X_train, X_test, y_train, y_test, X_train_res, Y_train_res

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns; sns.set()
import warnings
warnings.filterwarnings("ignore")

#1. smote전 + ls
#2. smote전 + rf
#3. smote후 + ls
#4. smote후 + rf

def logisticR(X_train, y_train, X_test, y_test, disc):
    lr = LogisticRegression()
    lr.fit(X_train, y_train.ravel())   #resample 전 모델 학습
    y_test_pre = lr.predict(X_test)

    print("\n"+disc + "accuracy_score: :{:.2f}%".format(accuracy_score(y_test, y_test_pre)*100))
    print(disc + "recall_score: :{:.2f}%".format(recall_score(y_test, y_test_pre) * 100))
    print(disc + "precision_score: :{:.2f}%".format(precision_score(y_test, y_test_pre) * 100))
    print(disc + "roc_auc_score: :{:.2f}%".format(roc_auc_score(y_test, y_test_pre) * 100))


    cnf_matrix = confusion_matrix(y_test, y_test_pre)
    print(disc + " ===>\n", cnf_matrix)  #matrix 개수
    print("cnf_matrix_test[0,0]: \n", cnf_matrix[0,0])
    print("cnf_matrix_test[0,1]: \n", cnf_matrix[0,1])
    print("cnf_matrix_test[1,0]: \n", cnf_matrix[1,0])
    print("cnf_matrix_test[1,1]: \n", cnf_matrix[1,1])


    print(disc+" matrix_accuracy_score : ", (cnf_matrix[1, 1] + cnf_matrix[0, 0]) / (cnf_matrix[1, 0] + cnf_matrix[1, 1] + cnf_matrix[0, 1] + cnf_matrix[0, 0]) * 100)


    print(disc+" matrix_recall_score : ",
          cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]) * 100)

def rf(X_train, y_train, X_test, y_test, disc):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train.ravel()) #reshape 한 모델
    y_test_pre = rf.predict(X_test)

    cnf_matrix_rf = confusion_matrix(y_test, y_test_pre)
    print(disc+" matrix_accuracy_score: ", (cnf_matrix_rf[1, 1] + cnf_matrix_rf[0, 0]) / (cnf_matrix_rf[1, 0] + cnf_matrix_rf[1, 1] + cnf_matrix_rf[0, 1] + cnf_matrix_rf[0, 0]) * 100)

    print(disc+" matrix_recall_score : ", (cnf_matrix_rf[1, 1] / (cnf_matrix_rf[1, 0] + cnf_matrix_rf[1, 1]) * 100))




if __name__ == "__main__":
    X_train = creditcard.X_train   #creditcard.py 파일에서 만든 X_train을 가져와서 X_train으로 저장
    y_train = creditcard.y_train
    X_test = creditcard.X_test
    y_test = creditcard.y_test

    X_smote = creditcard.X_train_res
    y_smote = creditcard.y_train_res

    logisticR(X_train, y_train, X_test, y_test, "smote전 + logisticR") #smote 전
    logisticR(X_smote, y_smote, X_test, y_test, "smote후 + logisticR") #smote 후
    rf(X_train, y_train, X_test, y_test, "smote전 + RF")#smote 전
    rf(X_smote, y_smote, X_test, y_test, "smote후 + RF") #smote 후

