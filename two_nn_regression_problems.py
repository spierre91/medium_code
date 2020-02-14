"""
Created on Wed Feb 12 12:48:54 2020

@author: sadrachpierre
"""

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


df = pd.read_csv("cars.csv")
print(df.head())
X = np.array(df[['age', 'gender', 'miles', 'debt', 'income']])
y = np.array(df['sales'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

y_train=np.reshape(y_train, (-1,1))

print(df.head())

model = Sequential()
model.add(Dense(32, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'], validation_split = 0.2)
model.fit(X_train, y_train, epochs=100, batch_size=10 )

y_pred = model.predict(X_test)



import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')


df = pd.read_csv("auto-mpg.csv")
df.dropna(inplace = True)
del df['Unnamed: 0']
print(df.head())
X = np.array(df[['Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']])
y = np.array(df['MPG'])


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

y_train=np.reshape(y_train, (-1,1))


model = Sequential()
model.add(Dense(64, input_dim=7, kernel_initializer='normal', activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'], validation_split = 0.2)
model.fit(X_train, y_train, epochs=1000, batch_size =10)
y_pred = model.predict(X_test)

import matplotlib.pyplot as plt 
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Prediction')
