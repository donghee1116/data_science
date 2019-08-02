
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()
print(type(iris))

kf_data = iris.keys()
print("<< kf_data >>")
print(kf_data)

kf_data = iris.data
kf_label = iris.target
kf_columns = iris.feature_names


#alt + shift + E
kf_data = pd.DataFrame(kf_data, columns=kf_columns)
print("<< kf_label >>")
print(kf_label)
print("pd.value_counts(kf_label)\n", pd.value_counts(kf_label))
print("kf_label.sum()\n", kf_label.sum())
print("kf_label.dtype\n",kf_label.dtype)


def kfold():
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, random_state=0)

    #split() 은 학습용과 검증요의 데이터 인덱스 출력
    for i, (train_idx, valid_idx) in enumerate (kf.split([kf_data.values, kf_label])):
        train_data, train_label = kf_data.values[train_idx, :], kf_label[train_idx]
        valid_data, valid_label = kf_data.values[valid_idx, :], kf_label[valid_idx]

        print("{} Fold train label\n{}".format(i, train_label))
        print("{} Fold valid label\n{}".format(i, valid_label))
        #print("{} Fold train idx\n train label\n{}".format(i, train_idx, train_label))

def Stratified_KFold():
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=5, random_state=0)

    #split()은 학습용과 검증용의 데이터 인덱스 출력
    for i, (train_idx, valid_idx) in enumerate (kf.split([kf_data.values, kf_label])):
        train_data, train_label = kf_data.values[train_idx, :], kf_label[train_idx]
        valid_data, valid_label = kf_data.values[valid_idx, :], kf_label[valid_idx]

        print("{} Fold train label\n{}".format(i, train_label))
        print("{} Fold valid label\n{}".format(i, valid_label))
        #print("Cross Validation Score:{:.2f}$".format(np.mean(val_scores)))




















