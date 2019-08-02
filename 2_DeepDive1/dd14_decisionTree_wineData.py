from sklearn.datasets import load_wine
wine = load_wine()
#print(wine.DESCR)

from sklearn.datasets import load_wine
#from util.logfile import logger
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import  matplotlib.pyplot as plt

#데이터 호출
wine = load_wine()
data = wine.data
label = wine.target
#columns = wine.feature_names
#logger.debug(wine.DESCR)
#data = pd.DataFrame(Data, columns = columns
#logger.debug(data.info())

x_train, x_test, y_train, y_test = train_test_split(data, label, stratify = label, random_state=0)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train, y_train)
score_tr = tree.score(x_train, y_train)
score_te = tree.score(x_test, y_test)
print('DT 훈련 세트 정확도: \n', format(score_tr))
print('DT 테스트 세트 정확도: \n', format(score_te))



#pre-pruning 의 방법 중 하나는 깊이의 최대를 설정
tree1 = DecisionTreeClassifier(max_depth=2, random_state=0)
tree1.fit(x_train, y_train)
score_tr1 = tree1.score(x_train,y_train)
score_te1 = tree1.score(x_test, y_test)
print('DT 훈련 depth 세트 정확도: \n', format(score_tr1))
print('DT 테스트 depth 세트 정확도: \n', format(score_te1))


#tree module 의 export graphviz 함수를 이용해 tree 시각화
import graphviz

from sklearn.tree import export_graphviz
export_graphviz(tree1, out_file='tree1.dot', class_names=wine.target_names, impurity=False, filled=True) #impurity: gini 미출력, filled: node 색깔 다르게

with open('tree1.dot') as file_reader:
    dot_graph = file_reader.read()

dot = graphviz.Source(dot_graph) #dot_graph의 source저장
dot.render(filename='tree1.png') #png로 저장
#https://graphviz.gitlab.io/_pages/Download/Download_windows.html

#tree module의 export graphviz 함수를 이용해 tree시각화
print("특성 중요도 첫번째 : \n", format(tree.feature_importances_))

print("특성 중요도 두번째 : \n", format(tree1.feature_importances_))

print("\nwine.data.shape : \n",wine.data.shape)
n_feature = wine.data.shape[1]
print(n_feature)

idx = np.arange(n_feature)
print('idx: \n', idx)

feature_imp = tree.feature_importances_
plt.barh(idx, feature_imp, align= 'center')
plt.yticks(idx, wine.feature_names)
plt.xlabel('feature importance', size = 15)
plt.ylabel('feature', size = 15)

plt.show()





















