import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("46\weight-height.csv")

X = data['Height'].values
Y = data['Weight'].values

X_train , Y_train , X_test , Y_test = train_test_split(X,Y , shuffle=True , test_size=0.2)

print(X_train.shape)

X_train = X_train.reshape(-1,1)
Y_train = Y_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)

print(X_train.shape)