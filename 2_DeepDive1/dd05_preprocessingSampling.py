from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

#1. 모델 불러오기 및 정의
ros = RandomOverSampler(random_state=2019) #data를 어디서 짜를지 말해주는것
rus = RandomUnderSampler(random_state=2019)


#2. fit
#데이터에서 특징을 합습함과 동시에 데이터 샘플링

#over sampling
oversampled_data, oversampled_label = ros.fit_resample(data, label)
oversampled_data = pd.DataFrame(oversampled_data, columns = data.columns)

#under sampling
undersampled_data, undersampled_label = rus.fit_resample(data, label)
undersampled_data = pd.DataFrame(undersampled_data, columns = data.columns)


#3. Transform
print('원본 데이터의 클래스 비율 \n {}'.format(pd.get_dummies(label).sum()))
print('\n Random Over Sampling result: \n{}'.format(pd.get_dummies(oversampled_label).sum()))
print('\n Random Under Sampling result: \n{}'.format(pd.get_dummies(undersampled_label).sum()))

#4. 결과 확인





###################################
#Random Over, Under Sampling 코드 실행 예제

def psampling(data):
    #print("sampling")
    label = data['Sex']
    ros = RandomOverSampler()
    rus = RandomUnderSampler()
    oversampled_data, oversampled_label = ros.fit_resample(data, label)
    undersampled_data, undersampled_label = rus.fit_resample((data, label))

    oversampled_data = pd.DataFrame(oversampled_data, columns = data.columns)
    undersampled_data = pd.DataFrame(undersampled_data, columns = data.columns)
    print('원본 데이터의 클래스 비율 \n {}'.format(pd.get_dummies(label).sum()))
    print('\n Random Over Sampling result: \n{}'.format(pd.get_dummies(oversampled_label).sum()))
    print('\n Random Under Sampling result: \n{}'.format(pd.get_dummies(undersampled_label).sum()))

if __name__ == '__main__':
    psampling(importData())













