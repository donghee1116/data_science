

import matplotlib.pyplot as plt

#기본 선 그래프
#plot(x축 데이터, y축 데이터), plot 함수를 이용해서 x , y 축 데이터를 기준으로 그래프를 그려준다
# show()함수를 이용하여 plot 에서 그려준 그래프를 출력한다

#plt.plot([1,2,3,4], [1,4,9,16])
#plt.show()


import numpy as np
#
# X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
# C, S = np.cos(X), np.sin(X)

# pt1 = plt.plot(X, C)
# pt2 = plt.plot(X, S)
#print(pt1, pt2)

# plt.show()


#line color 설정
# color 인자를 사용하여 그래프 선 색 결정 가능
#line color의 default 값은 blue, Line color를 gray로 표시할 때는 '0~1' 사이의 값으로 세팅
#
# years = [x for x in range(1950, 2011,10)]
# gdp = [y for y in np.random.randint(300, 10000,size=7)]

# plt.plot(years, gdp, color = '0.7')
# plt.show()

# #line style 설정
# t = np.arange(0,5,0.2)
# a = plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.show()
# print(a)





#line width 설정
#
# x = [1,2,3,4,5]
# y = [1,4,9,16,25]
# plt.plot(x,y, linewidth= 3.0)
# plt.show()



#line marker 설정 1
#marker 인자를 사용하여 그래프 마커 스타일 설정 가능
#markersize: 마커 크기 markeredgewidth: 마커 가장자리 굵기, markeredgewidthcolor: 마커 가장자리 색, markerfacecolor :마커 색
# years = [x for x in range(1950, 2011,10)]
# gdp = [y for y in np.random.randint(300, 10000,size=7)]
# plt.plot(years, gdp, marker = 'h', markersize = 6, markeredgewidth = 1, markeredgecolor = 'red', markerfacecolor = 'black')
# plt.show()



#line marker 설정 2
#마커 인자를 사용하여 그래프 마커스타일 설정 가능
#
# plt.plot([1,2,3,4], 'rs')
# plt.show()

#line label/ legend 설정
#label 인자를 사용하여 선의 라벨 지정, legend함수 처리를 위해 지정 필요
#legend()함수 : plot의 label에 의해 범례 표시

# a = b = np.arange(0,3,0.2)
# print(a)
# c=np.exp(a)
# print(c)
# d=c[::-1] #c의 순서를 뒤에서 앞으로 바꿔줌
# print(d)

# plt.plot(a, c, 'k--', label = 'Model Length')
# plt.plot(a,d,'k:', label = 'Data length')
# plt.plot(a, c+d, 'k', label = 'Total Message length')
# plt.legend() #legend 설정
# plt.show()

# #plot.axis(xmin, xmax, ymin, ymax)

# a= plt.plot([1,2,3,4],[1,4,9,16], 'ro')
# b= plt.axis([0,6,0,20])
# plt.show()
# print(a)
# print(type(b), b)


#
# #Line xlabel, ylabel, xticks, xticks 설정
# #xlabel 함수 사용하여 x축 그래프에 의미 부여
# #ylabel 함수 사용하여 y축 그래프에 의미 부여
# #xticks 함수 사용하여 x축 세부 값 부여 xticks(ticks, labels) #ticks = 점
# #xticks 함수 사용하여 y축 세부 값 부여
#
# N = 5
# menMeans = (20, 35, 30, 35,27)
# width = 0.01
# ind = np.arange(N)
# print(ind)
# plt.bar(ind, menMeans)
# plt.title('Scores by group and gender')
# plt.ylabel("Scores")
# plt.xlabel("The number of people")
# print("ind+width/2.",ind+width/2.)
# plt.xticks(ind+width/2., ("G1",'G2','G3','G4','G5'))
# plt.yticks(np.arange(0,81,10))
# plt.legend(("Men"))
# plt.show()






#
# #line xlim/ylim 설정
# #xlim, ylim 함수를 사용하여 x,y축 내 범위 값 부여
#
# x = [1,2,3,4,5]
# y = [1,4,9,16,25]
# plt.plot(x,y)
# plt.title('Plot of y sv x')
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.xlim(0.0,7.0)
# plt.ylim(0.0, 30.0)
# plt.show()






# #기본 scatter plot 그래프
# #scatter (x axis data, y axis data), scatter 함수를 이용하여 x축 y축 데이터를 기준으로 산점도 그래프 그림
#
# data = np.random.rand(10,2)
# print(data)
# print(data.shape)
#
# plt.scatter(data[:,0], data[:,1])
# plt.show()
#





# #scatter 그래프의 모양과 색 입히기
# #size : 점 크기 (s 약자 사용 가능 )
# #color : 점 색상 (c 약자)
# #marker : 마커 종류
#
# x = np.random.rand(10)
# print(x)
# y = np.random.rand(10)
# print(y)
# z=np.sqrt(x**2+y**2)
# print(z)
#
# plt.subplot(321)
# plt.scatter(x, y, s = 80, c = z, marker= '>')
# plt.show()
#


# #기본 막대 그래프
# #bar(x axis data, y axis data), bar 함수를 이용해 x, y축 데이터를 기준으로 막대 그래프 그리기
# # 폭: 0.8, 색상은 파란색막대가 default.
#
# N = 5
# menMeans = (20,35,30,35,27)
# width = 0.01
# ind = np.arange(N)
# print(ind)
# plt.bar(ind, menMeans)
# plt.show()




#
# #barh 수평 막대 그래프
# #barh 함수를 사용하여 수평 막대 그래프 생성 가능
# #다중 수평 막대 그래프 생성 시 반대방향 데이터에 minus (-)를 부여
#
# w_pop = np.array([5,30,45,40], dtype=np.float32)
# m_pop = np.array([4,28,40,35], dtype=np.float32)
# x=np.arange(4)
# a = plt.barh(x, w_pop, color = 'r')
# b = plt.barh(x, -m_pop)
# print(a)
# print(b)
# plt.show()





# #pie graph --- pie(data)
# data = [5,10,30,20,7,8,10,10]
# plt.pie(data)
# plt.show()






#
# #Histogram  ----- hist(data)
# data = np.random.normal(5.0,3.0,1000)
# #plt.hist(data)
# plt.hist(data, bins = 15, facecolor = 'red', alpha = 0.4)
# plt.xlabel('data')
# plt.show()


# #file 읽고 그래프 그리기
# #파일을 읽고 plot함수를 통해 그래프 그리기
# X,Y = [],[]
# for line in open('data.txt','r'):
#     values = [float for s in line.split()]
#     X.append(values[0])
#     Y.append(values[1])
# plt.plot(X,Y)
# plt.show()











