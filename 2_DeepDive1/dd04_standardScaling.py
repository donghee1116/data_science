#p.20

import os
from os.path import join
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def importData():
    abalone_path = join('../data', 'abalone.txt')
    column_path = join('../data','abalone_attributes.txt')

    #print(abalone_path)
    #print(column_path)

    abalone_columns = list()
    for i in open(column_path):
        abalone_columns.append(i.strip())
        #print("abalone_columns: \n ", abalone_columns)
    data = pd.read_csv(abalone_path, header=None, names = abalone_columns)
    print("data.shape: \n", data.shape)
    #print("data.describe() \n", data.describe())
    print("data.info(): \n", data.info())
    label = data['Sex']
    del data['Sex']
    return data



def fsdscaler(data):
    sdscaler = StandardScaler()
    sdscaler.fit(data)  #평균, 표준편차값 찾기
    sdscaler_data = sdscaler.transform(data)
    sdscaler_pd = pd.DataFrame(sdscaler_data, columns=data.columns)
    #print("sdscaler_data => \n", sdscaler_pd)


if __name__ == '__main__':
    #importData()
    #mmscaler(importData())
    fsdscaler(importData())


