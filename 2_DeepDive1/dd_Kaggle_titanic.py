

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#acquire data
train_df = pd.read_csv('../data/titanicK_train.csv')
print(train_df)
#print(train_df.columns)
print(train_df.columns.values)


test_df = pd.read_csv('../data/titanicK_test.csv')
print(test_df)

combine = [train_df, test_df] #combine these datasets to run certain operations on both datasets together.


# preview the data
print("train_df.head()\n",train_df.head())

print("train_df.tail()\n", train_df.tail())

#to check variables's feature such as str, int, float. string is an object. "dtypes" 보면 됨
print("test_df.info()", train_df.info())
print('_'*40)
print("test_df.info()", test_df.info())


print("train_df.describe()\n", train_df.describe())


print("train_df.describe(include=['O'])\n", train_df.describe(include=['O']))  #여기서 나타내는게 뭐야?????
print()












