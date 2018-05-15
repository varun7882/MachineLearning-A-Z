# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
print "\n\nOur dataset :"
print dataset.head()
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values
print "\nFeatures are :"
print X
print "\nResult :"
print y

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print "\n\n\n\nAfter Taking care of missing data :"
print "\n\nOur dataset :"
print dataset.head()
print "\nFeatures are :"
print X
print "\nResult :"
print y

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print "\n\n\n\nAfter Encoding data :"
print "\nFeatures are :"
print X
print "\nResult :"
print y
