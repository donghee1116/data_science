#차원을 축소함

from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

#내장 데이터 불러오기
digits = load_digits()
#print("digits:\n",digits)
print("digits.keys() : \n", digits.keys())  #digits의 데이터가 dictionary 형태?? 어떻게 알아???

data = digits.data
print("data:\n",data)

label = digits.target
#print(label)


#데이터 시각화
plt.imshow(data[0].reshape(8,8))  #왜 8,8 만 되는거지????
plt.show()
print('Label: '.format(label[0]))  #뭘 위한것....?


#차원 축소
pca = PCA(n_components= 2)  #2차원에서 하겠단 소리?
pca.fit(data)
new_data = pca.transform(data)     #new_data 어떻게 생겼는지 어떻게 봄? type이 뭐임?

print("원본 데이터의 차원 : \n", format(data.shape))    #64차원
print("\nPCA 이후 데이터의 차원 : \n", format(new_data.shape))   #2차원 - n_components에 의해 결정되는것!!!!!


print("new_data[:,0]: \n", new_data[:,0]) #뭘 의미하는 숫자들임

#차원 축소 후 시각화
#plt.imshow(new_data[0]) #왜 여기서는 imshow 못써? 이건 다차원일때만 가능 ???
plt.scatter(new_data[:,0], new_data[:,1], c=label, edgecolors='black') #어떻게 data들의 색깔이 다 다름?
plt.show()
















