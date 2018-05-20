import numpy as np
import pandas as pd
dataset=pd.read_csv("iris.data.csv")
X=dataset.iloc[:,0:4].values
y=dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
