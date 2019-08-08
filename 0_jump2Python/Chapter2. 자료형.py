#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#to get multiple outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# # 1. 숫자형 (number)
# 
# 숫자 형태로 이루어진 자료형.
# 
#     정수형 integer    (ex. -3, 0, 2)
#     실수형 float      (ex. -12.231, 1.23e-10, 213.2E10)
#     8진수 oxtal       
#     16진수 hexadecimal

# In[7]:


# x 의 y 제곱 연산자 **
3**2


# In[9]:


#나 눗셈 후 몫을 반환하는 연산자 //
3//7


# In[ ]:


# 나눗셈 후 나머지를 반환하는 연산자 %


# In[26]:


7 % 3 
3 % 7


# # 2. 문자열 자료형 (string) - immutable 
# 문자, 단어 등으로 구성된 문자들의 집합을 의미. 
# 따옴표("", '', """ """, ''' ''')로 둘러싸여 있으면 모두 문자열!
# 
# 
# 

# In[29]:


# 문자열 더해서 연결하기 (Concatenation)
head = "Python"
tail = " is fun!"
head+tail
head,tail


# In[17]:


# 문자열 곱하기
a = 'python'
a*2 #여기서의 *는 곱하기가 아닌 반복하라는 의미!


# In[18]:


# 문자열 길이 구하기
a = "Life is too short"
len(a)


# In[37]:


## 문자열 인덱싱(indexing)과 슬라이싱(slicing) - 리스트나 튜플에서도 사용가능 
# 파이썬은 숫자를 0부터 셈!
# slicing a[시작번호: 끝번호] :  시작번호 <= a <끝번호. (끝번호 포함 안됨!!!)

a = "life is too short, you need python"
a[13]
a[0:4]
a[5:7]
a[:17]
a[-8]
a[19:-7]
a[-7:]


# ##        문자열 formatting
# 
# 문자열 안에 어떤 값을 삽입하는 방법
# 
# -------문자열 포멧 코드-------
# %s: 문자열 string  ----> 어떤 형태의 값이든 string으로 변환해 넣을 수 있다. 
# %c: 문자 1개 character
# %d: 정수 integer
# %f: 부동 소수 float
# %o: 8진수 oxtal
# %x: 16진수 hexadecimal
# %%: 문자 '%' 자체
# 

# In[39]:


# 1. 숫자 바로 대입

"I eat %d apples." % 3


# In[40]:


# 2. 문자열 바로 대입

"I eat %s apples." % "five"


# In[42]:


# 3. 숫자 값을 나타내는 변수로 대입

number = 3
"I eat %d apples." % number


# In[43]:


# 4. 2개 이상의 값 넣기   --- % 다음 괄호 안에 콤마(,)로 구분하여 각각의 값을 넣어주면 됨

number = 10
day = "three"

"I ate %d apples. So I was sick for %s days." % (number, day)


# In[46]:


# 5. %를 쓰려면 %%를 써야함. 
# "Error is %d%." %98
"Error is %d%%." %98


# ## 포맷 코드와 숫자 함께 사용하기

# In[53]:


# 1. 정렬과 공백

"%10s" % "hi" # 전체 길이가 10개인 문자열 공간에 대입되는 값을 오른쪽으로 정렬후, 그 앞은 남겨 놓아라

"%-10s" % "hi"# 왼쪽 정렬은 마이너스 

"%-10s jane" % "hi"# hi는 왼쪽 정렬은 마이너스하고 나머지 8은 공백. jane은 그 뒤에


# In[61]:


# 2. 소수점 표현하기 

"%.2f" %3.42342231 #'.'은 소숫점 포인트를 말하고, 그 뒤의 숫자 2는 소수점 뒤에 나올 개수를 말함. 

"%10.4f" % 213.23123124 #전체 길이가 10개인 문자열 공간에서 오른쪽으로 정렬, 소숫점 4자리 까지 나타내라. 


# ## format 함수를 사용한 formatting

# In[72]:


# 1. 숫자 바로 대입하기 

"I eat {} apples".format(3)

# 2. 문자열 바로 대입하기 

"I eat {} apples".format("five")

# 3. 숫자 값을 가진 변수로 대입하기 

number = 2
"I eat {} apples".format(number)

# 4. 2개 이상의 값 넣기 

num = 10
day = "three"

"I ate {} apples. So I was sick for {} days".format(num, day)

# 5. 이름으로 넣기 

"I ate {num} apples. So I was sick for {day} days".format(num = 7, day = "two")

# 6. 인덱스와 이름을 혼용해서 넣기 

"I ate {0} apples. so I was sick for {day} days".format(10,day = 3)


# ## format 함수를 이용한 정렬

# In[86]:


# 1. 왼쪽 정렬

"{0:<10}".format("hi") # 문자열의 총 자리수는 10, 왼쪽으로 정렬

"{0:>10}".format("hi") # 문자열의 총 자리수는 10, 오른쪽으로 정렬

"{0:^10}".format("hi") # 문자열의 총 자리수는 10, 가운데로 정렬

# 2. 공백 채우기 

"{0:=^10}".format("hi") 

"{0:!<10}".format("hi") 

# 3. 소수점 표현하기 

y = 3.141592
"{0:0.2f}".format(y)

"{:10.2f}".format(y)

# 4. {또는} 문자 표현하기 

"{{and}}".format()


# ## 문자열 관련 함수 iterable, immutable
# 
# 문자열 자료형은 자체적으로 함수를 가지고 있음 aka. 내장함수.
# 이 내장 함수를 사용하려면 문자열 변수 이름 뒤에 '.'를 붙인 다음 함수 이름을 써주면 됨. 
# ex."abc".count('b')
# 

# In[96]:


# 1. 문자 수 세기 (count())

a = "hobby"
a.count('b')


# In[97]:


# 2. 위치 알려주기 I (find())

a = "Python is the best choice"
a.find('b') # 문자열에서 b가 처음 나온 위치
a.find('k')  #문자열이 존재하지 않을 시, -1 반환


# In[102]:


# 3. 위치 알려주기 II (index())

a = "Life is too short"
a.index('t')

# a.index('k')
try:
    a.index('k')  #index는 finde와 다르게 없으면 오류남 
except ValueError:
    print(-1)


# In[108]:


# 4. 문자열 삽입 (join()) - 리스트, 튜플에도 사용 가능

"-".join('abcd')

#리스트

"-".join(['a','b','c','d'])   #리스트를 join함수에 넣으면 string 반환함.


# In[112]:


# 5. 소문자를 대문자로 바꾸기 (upper())

a = 'hi'
a.upper()


# 6. 대문자를 소문자로 바꾸기 (lower())

b = 'BYE'
a.lower()


# In[115]:


# 7. 왼쪽 공백 지우기 (lstrip())

a = '    hi      '
a.lstrip()

# 8. 왼쪽 공백 지우기 (rstrip())

a = '    hi      '
a.rstrip()

# 9. 양쪽 공백 지우기 (strip())

a = '    hi      '
a.strip()


# In[118]:


# 10. 문자열 바꾸기 (replace())

a = 'Life is too short'
a.replace("Life","Your leg")


# In[119]:


# 11. 문자열 나누기 (split())

a = 'Life is too short'
a.split() # 공백을 기준으로 문자열 나눔

b = "a:b:c:d"
b.split(":") # : 기준으로 문자열 나눔


# # 3. 리스트 자료형 iterable, mutable
# 리스트명 = [요소1, 요소2, 요소3, ...]
# 리스트 안에는 어떠한 자료형도 포함시킬 수 있음! 

# In[ ]:


#  여러 리스트의 생김새

a = []       # 비어있는 리스트 생성방법1
aa = list()  # 비어있는 리스트 생성방법2
b =[1,2,3]
c = ['Life', 'is', 'too', 'short']
d = [1,2, 'Life', 'is']   #숫자와 문자열을 함께 요소값으로 가질 수도. 
e = [1,2,['Life','is']]  #리스트 자체를 요소값으로 가질 수도!


# In[132]:


#리스트의 인덱싱(indexing) 

a = [1,2,3]
a[0]
a[0]+a[2]

b = [1,2,3,['a','b','c']]
b[-1]
b[-1][0]
b[-1][0] + str(b[1])


# In[137]:


#리스트의 슬라이싱(slicing) 

a = [1,2,3,4,5]
a[0:3] # a[3]은 포함이 안됨!!!
a[3:]


#문자열에서도 가능했음
a = '12345'
a[0:3]

a = [1,2,3,['a','b','c'],4,5]
a[3][:2]


# In[143]:


# 리스트 연산하기 (+로 더하기, *로 반복하기)

# 1. 리스트 더하기 (+)
a = [1,2,3]
b = [4,5,6]

a+b

# 2. 리스트 반복하기 (*)
a*3

# 문자열과 비교
c = 'abc'
c*3

# 3. 리스트 길이 구하기 (len)

a = [1,2,3]
len(a)


# In[147]:


# 4. 리스트 값 수정하기 

a = [1,2,3]
a[2] = 4
a

# 5. del 함수 이용해서 리스트 요소 삭제하기 (del listName[index])  -- pop(), remove()로도 가능. (밑에 나옴)
a = [1,2,3]
del a[1]
a

b = [1,2,3,4,5,6]
del b[3:]
b


# ## 리스트 관련 함수 
# 위의 del은 파이썬의 내장 함수. 밑에 나오는 .을 사용하는 함수는 클래스의 매서드 느낌. (. 은 참조를 뜻함) 

# In[153]:


# 1. 리스트에 요소 추가 (append()) - list의 맨 마지막에 x를 추가하는 함수 

a = [1,2,3]
a.append(4)
a

a.append([4,5])   #리스트 안에는 어떤 자료형도 추가 가능
a
a.append('hi')
a


# In[155]:


# 2. 리스트 정렬 (sort()) -- 순서대로 정렬해줌

a = [32,1,4,5]
a.sort()
a

a = ['c','a','l','z','g']
a.sort()
a


# In[156]:


# 3. 리스트 뒤집기 (reverse()) - 리스트를 역순으로 뒤집어 줌 
a = [32,1,4,5]
a.reverse()
a


# In[158]:


# 4. 위치 반환 (index(x) : 리스트에 x값이 있으면 x의 위치 값을 돌려준다) 

a = [1,2,3,'a']
a.index(3)
# a.index(6) #없는거 쓰면 오류남 
a.index('a')


# In[162]:


# 5. 리스트에 요소 삽입 (insert(a,b) : a번째 위치에 b를 삽입)

a = [1,2,3]
a.insert(1,'a')
a
# a.insert('n') # insert는 2개의 arguments를 요구해서 한개만 쓰면 오류남 


# In[163]:


# 6. 리스트 요소 제거 (remove(x) : 리스트에서 첫 번째로 나오는 x를 삭제하는 함수)

a = [1,2,3,1,2,3]
a.remove(3)
a


# In[166]:


# 7. 리스트 요소 끄집어내기 
#pop(): 리스트의 맨 마지막 요소를 돌려주고 그 요소는 삭제. 
#pop(x): 리스트의 x번째 요소를 돌려주고 삭제

a = [1,2,3]
a.pop()  #pop(): 리스트의 맨 마지막 요소를 돌려주고 그 요소는 삭제. 
a

b = [1,2,3,4,5]
b.pop(2)  #pop(x): 리스트의 x번째 요소를 돌려주고 삭제
b


# In[167]:


# 8. 리스트에 포함된 요소 x의 개수 세기 (count(x) : 리스트 안에 x가 몇 개 있는지 조사하여 그 개수를 돌려주는 함수)

a = [1,1,2,2,2,2,2,2,3,3,1,3,2]
a.count(1)


# In[173]:


# 9. 리스트 확장 (extend(x) : x에는 리스트만 올 수 있으며, 원래의 리스트 a에 x리스트를 더하게 됨)
 
a = [1,2,3]
a.extend([4,5])  # a += [4,5] 와 동일!
a
a.extend([['d'],'a','b',3])
a


# # 4. 튜플 자료형 iterable, immutable
# 튜플과 리스트는 거의 유사하나, 다음 2가지가 다름.
# 1. list 은 [] 으로 둘러쌈. 하지만, tuple 은 () 으로 둘러 쌈
# 2. list 는 그 값의 생성, 삭제, 수정이 가능하지만 (mutable), tuple은 값을 바꿀 수 없음 (immutable).
# 
# 프로그램이 실행되는 동안 그 값이 항상 변하지 않기르 바란다거나, 값이 바뀔까 걱정하고 싶지 않다면, 주저 말고 튜플을사용해야. 
# 
# 실제 프로그램에서는 값이 변경되는 형태의 변수가 훨씬 많기 때문에 평균적으로 튜플보다는 리스트를 더 많이 사용함. 

# In[175]:


# 여러 튜플의 생김새 
t1 = ()
t2 = (1,)    # 1개의 요소만 있을 때, 요소 뒤에 콤마를 반드시 붙여야 함!!
t3 = (1,2,3)
t4 = 1,2,3   #괄호를 생략해도 무방!! - 알아서 생김. 
t5 = ('a','b','c',('ab','cd'))

t1
t2
t3
t4
t5


# In[179]:


# 1. tuple indexing
t1 = (1,2,"a",'b')
t1
t1[0]
t1[3]


# In[181]:


# 2. tuple slicing 
t1 = (1,2,'a','b')
t1[:1]
t1[1:]


# In[182]:


# 3. 튜플 합치기 (+)
t1 = (1,2,'a','b')
t2 = (3,4)
t1+t2


# In[183]:


# 4. 튜플 반복하기 (*)
t1 = (1,2,'a','b')
t1*3


# In[184]:


# 5. 튜플 길이 구하기 (len)
t1 = (1,2,'a','b')
len(t1)


# In[185]:


#(1,2,3) 이라는 튜플에 값 4를 추가하여 (1,2,3,4)를 만들어 출력해보기 

t1 = (1,2,3)
t1+ (4,)    #이거는 튜플을 수정하는게 아니라 튜플들을 합친것! 


# # 5. 딕셔너리 자료형 mutable, iterable, not sequencial  
# 
# '이름' = '홍길동', '생일'=yyyymmdd 처럼 대응관계를 나타낼 수 있는 자료형을 '연관 배열' (associative array) 또는 '해시'(hash)라고 부름
# 
# 파이썬에서는 이런 자료형을 Dictionary라 칭함. - key와 value를 한 쌍으로 갖는 자료형
# 
# dic = {key1:value1, key2:value2, key3:value3, ... }
# 
# 인덱싱 방법 적용 불가!
# 
# **주의사항**
# 1. Key는 고유한 값이므로 중복된 Key 값을 설정해 놓으면 하나를 제외한 나머지 것들은 무시됨.
# 2. Key에는 list를 쓸 수 없음. 대신, tuple은 가능. Value는 아무거나 와도 상관 없음. 

# In[214]:


#빈 dictionary 생성 방법
a = dict()
a

a = {}
a

dic = {'name':'dk','phone':'01012345678','birth':1111}
dic



a = {1:'hi'}
a
b = {'a':[1,2,3]} #value에 list넣을 수도 있음
b


# In[197]:


# 1. 딕셔너리 쌍 추가하기 : dicName[keyName] = value

a = {1: 'a'}
a[2] = 'b'    #key 가 2, value가 b 인 쌍을 추가
a
a['가'] = 3     #key 가 '가', value가 3 인 쌍을 추가
a
a['name'] = 'mike'
a[4] = [1,2,3]
a


# In[202]:


# 2. 딕셔너리 요소 삭제하기 : del dicName[keyName]

a = {1: 'a', 2: 'b', '가': 3, 'name': 'mike', 4: [1, 2, 3]}
del a[1] # key가 1인 key:value 쌍 삭제 
a
del a['name']   # key가 'name'인 key:value 쌍 삭제 
a


# In[218]:


# 3. 딕셔너리에서 key 사용해 value얻기 : dicName[keyName] --> value 반환

grade = {'pey':10, 'julliet':99}
grade['pey']

#없는 key 를 사용하면 오류남 ----> get 함수 쓰면 됨 
# grade['name']
print(grade.get('name'))  # --> None 반환


# ## dictionary 관련 함수 

# In[205]:


# 1. Key 리스트 만들기 (a.keys())


a = {1: 'a', 2: 'b', '가': 3, 'name': 'mike', 4: [1, 2, 3]}
a.keys()   # a 의 key를 모아서 dic_keys 객체를 돌려줌 

# dic_keys의 객체를 리스트로 변환하려면 다음과 같이 해야 함. 
list(a.keys())


# In[207]:


# 2. Values 리스트 만들기 (a.values())

a = {1: 'a', 2: 'b', '가': 3, 'name': 'mike', 4: [1, 2, 3]}
a.values()

list(a.values())


# In[210]:


# 3. Key, Value 쌍 얻기 (items())

a = {1: 'a', 2: 'b', '가': 3, 'name': 'mike', 4: [1, 2, 3]}
a.items()   #Key, Value의 쌍을 튜플로 묶은 값을 dict_items 객체로 돌려줌. 


# In[213]:


# 4. Key, Value 쌍 모두 지우기 (clear)


a = {1: 'a', 2: 'b', '가': 3, 'name': 'mike', 4: [1, 2, 3]}
a.clear()
a


# In[219]:


# 5. Key로 Value 얻기 (get())

grade = {'pey':10, 'julliet':99}
#없는 key 를 사용하면 오류남 ----> get 함수 쓰면 됨 
# grade['name']
print(grade.get('name'))  # --> None 반환

# 딕셔너리 안에 찾으려는 Key 값이 없을 경우 None 대신 미리 정해 둔 디폴트 값을 대신 가져오게 하고 싶을 때 : get(keyName, 디폴프값)

grade.get('name','there is no key')


# In[221]:


# 6. 해당 Key가 딕셔너리 안에 있는지 조사하기 ( keyName in dicName)

a = {1: 'a', 2: 'b', '가': 3, 'name': 'mike', 4: [1, 2, 3]}
'name' in a

'foo' in a


# # 6. 집합 자료형 set(List or String)
# 
# 집합에 관련된 것을 쉽게 처리하기 위해 만든 자료형. 
# 
# ==집합 자료형의 특징==
# 
# 1. 중복을 허용하지 않는다. -> 자료형의 중복을 제거하기 위한 필터 역할로도 종종 사용
# 2. 순서가 없다.  -> indexing X

# In[229]:


s1 = set([1,2,3])    #set(List)
s1

s2 = set("Hello")     #set(String)
s2


# In[234]:


# 인덱싱이 불가는 한데, 인덱싱으로 접근하려면, 리스트나 튜플로 변환한 후 해야함. 

set1 = set([1,2,3])
set1
# set1[1]  # set은 인덱싱 불가 이므로 오류남 

list1 = list(s1) # list로 변환
list1
list1[1]

tuple1 = tuple(set1) #tuple로 변환
tuple1 
tuple1[1]


# In[245]:



set1 = set([1,2,3,4,5])
set2 = set([4,5,6,7,8])

set1
set2

# 교집합  (&), (intersection())
set1 & set2

set1.intersection(set2)

# 합집합 (|) , (union())- 중복된것은 한번씩만 표현됨
set1|set2

set1.union(set2)

# 차집합 (-), (difference())
set1 - set2
set1.difference(set2)
set2 - set1
set2.difference(set1)


# ## 집합 자료형 관련 함수

# In[246]:


# 1. 값 1개 추가하기 (add)

s1 = set([1,2,3])
s1.add(4)
s1


# In[247]:


# 2. 값 여러개 추가하기 (update)

s1 = set([1,2,3])
s1.update([4,5,6])
s1


# In[249]:


# 3. 특정 값 1개 제거하기 (remove)

s1 = set([1,2,3])
s1.remove(2)
s1


# # 7. 불 (bool) 자료형
# 
# 1. 참(True) 과 거짓(False)을 나타내는 자료형으로, True, False 2가지 값만 갖는다.
# 2. 조건문의 반환 값으로도 사용된다. 
# 3. 자료형의 참, 거짓을 나타 낼 때 쓰임

# In[251]:


a = True
b = False
type(a)
type(b)


# ## 자료형의 참 거짓
# 
# 
# 문자열, 리스트, 튜플, 딕셔너리 등의 
#                            
#                             값이 비어있는 경우: False
# 
#                               안 비어있는 경우: True
#                               
#                                          0:False
#                                0이 아닌 숫자 : True
#                                
#                                      None : False

# In[258]:


bool('python')
bool(' ')
bool('')
bool(123)
bool(0)
bool([])
bool([1,2,3])


# # 8. 자료형의 값을 저장하는 공간, 변수
# 
# 다른 언어들은 자료형을 직접 지정해야 하지만, 파이썬은 변수에 저장된 값을 스스로 판단하여 자료형을 지정! 
# 
# 변수: 객체를 가리키는 것
# 
# 객체: 자료형
# 
# ex. a = [1,2,3]
# 
# - [1,2,3]값을 가지는 리스트 자료형(객체)이 자동으로 메모리에 생성
# - 변수 a는 [1,2,3] 리스트가 저장된 메모리의 주소를 가리키게 됨 
# 
# 메모리: 컴퓨터가 프로그램에서 사용하는 데이터를 기억하는 공간

# In[259]:


# 변수가 가리키는 객체의 메모리의 주소 확인법: id(변수 이름)
a = [1,2,3]
id(a)


# In[262]:


# 만약 리스트를 복사한다면? 완전 똑같은 메모리 주소를 사용. 
#변하는 사실? [1,2,3] 리스트를 참조하는 변수가 a 한개에서 a,b 두개로 늘었다는 사실. 
a = [1,2,3]
b = a

id(a)
id(b)

a is b # is: 동일한 객체를 가리키고 있는지에 대해서 판단하는 파이썬 명령어

a[1] = 4 
a
b   # a를 바꾸니 b도 같이 바뀜 


# In[269]:


## b 변수 생성 시, a 변수의 값을 가져오되, a 와는 다른 주소를 가리키도록 만드려면? 

# 1. [:] 사용

a = [1,2,3]
b = a[:] # list의 모든 요소 slicing
id(a)
id(b)

a is b 

a[1] = 4
a
b


# In[268]:


# 2. copy 모듈 사용
from copy import copy
a = [1,2,3]
b = copy(a)
id(a)
id(b)

a[1] = 4
a
b

a is b


# ## 변수를 만드는 여러가지 방법

# In[272]:


# 1. 튜플로 a,b에 값 대입
a, b = ('python', 'life')
a
b
# type(a)
# type(b)

# 이 방법은 다음 예문과 완전 동일
(a, b) = 'python', 'life'   #튜플은 괄호 생략 가능하므로! 
a
b
# type(a)
# type(b)


# In[274]:


# 2. 리스트로 변수를 만들기 
[a,b] = ['python','life']
a
b
# type(a)
# type(b)


# In[275]:


# 3. 여러개 변수에 같은 값 대입
a = b = 'python'

a
b

