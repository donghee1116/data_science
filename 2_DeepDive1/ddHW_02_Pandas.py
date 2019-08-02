#Pandas: 축의 이름을 따라 데이터를 정렬할 수 있는 자료 구조
# 다양한 소스에서 가져온 다양한 방식으로 색인된 데이터를 핸들링 가능함
# 시계열 비시계열 데이터 모두 다룰 수 있는 자료 구조
# 누락된 데이터의 유연한 처리 및 SQL과 같은 연산 수행

# data frame 구조
# M*N 행렬 구조를 가지는 데이터 구조

import numpy as np
from scipy.stats import mode
import statistics as sta

x=np.array([-2,1,-1,1,1,4.3])
print(type(x))
print(np.mean(x))
print(x.shape)
print(mode(x))

























