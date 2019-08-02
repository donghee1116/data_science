#그냥 그래프들


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def data():
    data1 = np.random.randn(200)
    data2 = np.random.randn(200)+4
    data3 = np.random.randn(200)+8
    return data1, data2, data3

def ranPlt():
    a,b,c = data()
    plt.scatter(range(200), a, c='red')
    plt.show()
    data2=np.random.randn(200)+4
    plt.scatter(range(300,500), b, c = 'blue')
    plt.show()
    data3=np.random.randn(200)+8
    plt.scatter(range(600,800), c, c='green')
    plt.show()

if __name__ == "__main__":
    ranPlt()



#한 그래프에 모두 불러오기

def ranOnePlt(): #화면에 plot그리기
    a,b,c = data()
    plt.scatter(range(200), a, c = 'red')
    plt.scatter(range(300, 500), b, c='blue')
    plt.scatter(range(600, 800), c, c='green')
    plt.show()

if __name__ == "__main__":
    ranOnePlt()






def scipy_zscore():
    a, b, c = data()
    from scipy import stats
    a = stats.zscore(a)
    b = stats.zscore(b)
    c = stats.zscore(c)

    plt.scatter(range(200), a, c = 'red')
    plt.scatter(range(300, 500), b, c='blue')
    plt.scatter(range(600, 800), c, c='green')
    plt.show()

if __name__ == "__main__":
    scipy_zscore()


def sklearn_StandardScaler():
    from sklearn.preprocessing import StandardScaler
    a,b,c = data()
    data1 = pd.Series(a)
    data2 = pd.Series(b)
    data3 = pd.Series(c)
    mscaler = StandardScaler()
    mMscaled_a = mscaler.fit_transform(data1.values.reshape(-1,1))
    mMscaled_b = mscaler.fit_transform(data2.values.reshape(-1, 1))
    mMscaled_c = mscaler.fit_transform(data3.values.reshape(-1, 1))

    plt.scatter(range(200), mMscaled_a, c ='red')
    plt.scatter(range(300, 500), mMscaled_b, c='blue')
    plt.scatter(range(600, 800), mMscaled_c, c='green')
    plt.show()

if __name__ == "__main__":
    sklearn_StandardScaler()









