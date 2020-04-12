import pandas as pd

df = pd.read_csv("insurance.csv")

print(df.head())
df['sex_code'] = np.where(df['sex'] == 'female', 1, 0)
df['smoker_code'] = np.where(df['smoker'] == 'yes', 1, 0)
import numpy as np 
X = np.array(df[['age', 'bmi', 'children', 'sex_code', 'smoker_code']])
y = np.array(df['charges'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.linear_model import LinearRegression


reg = LinearRegression()
reg.fit(X_train, y_train)

print("Model Performance: ", reg.score(X_test, y_test))
