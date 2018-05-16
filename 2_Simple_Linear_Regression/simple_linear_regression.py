# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.33, random_state = 24)
'''print X_train
print X_test
print y_train
print y_test

print 'shapes are :'
print X_train.shape
print y_train.shape
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
y_train=y_train[:,0]
print 'shapes are :'
print X_train.shape
print y_train.shape'''
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
ytp=regressor.predict(X_train)
# Visualising the results
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
f.suptitle('All figures')
ax1.scatter(X_train, y_train, color = 'red')
ax1.set_title("Training Data")
ax2.scatter(X_train, y_train, color = 'red')
ax2.plot(X_train, ytp, color = 'blue')
ax2.set_title("Regression Line on Training Data")
ax3.scatter(X_test, y_test, color = 'red')
ax3.set_title("Test Data")
ax4.scatter(X_test, y_test, color = 'red')
ax4.plot(X_train, ytp, color = 'blue')
ax4.set_title("Regression line fittting on Test Data")
plt.show()
