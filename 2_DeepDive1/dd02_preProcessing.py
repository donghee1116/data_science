

import os
from os.path import join
import pandas as pd
from sklearn.preprocessing import MinMaxScaler #전처리





def importData():
    abalone_path = join('../data', 'abalone.txt')
    column_path = join('../data','abalone_attributes.txt')
    #print(abalone_path)
    #print(column_path)
    abalone_columns = list()
    for i in open(column_path):
        abalone_columns.append(i.strip())
    #print(abalone_columns)

    data = pd.read_csv(abalone_path, header = None, names = abalone_columns)
    #data.shape
    #print("describe=Wn", data.describe())
    #data.info()
    #label = data['Sex']
    #del data['Sex']
    return data

#cwd = os.getcwd()
#print(cwd)

importData()













