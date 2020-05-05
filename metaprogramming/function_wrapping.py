import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


import time
from functools import wraps

def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end-start)
        return result
    return wrapper

@timethis
def read_and_split(test_size):
    df = pd.read_csv("insurance.csv")
    print(df.head())
    X = np.array(df[['children', 'bmi', 'age' ]])
    y = np.array(df['charges'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = read_and_split(0.5)
 
@timethis
def fit_model():
    model = LinearRegression()
    model = model.fit(X_train, y_train)
    return model

model = fit_model()

@timethis
def predict():
    result = model.predict(X_test)
    return result

prediction =  predict()
