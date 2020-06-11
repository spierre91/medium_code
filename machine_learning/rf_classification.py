import pandas as pd 
import numpy as np 

df = pd.read_csv("mushrooms.csv")

print("Shape: ", df.shape)

print(list(df.columns))

print(df.head())

df_cat = pd.DataFrame()
for i in list(df.columns):
    df_cat['{}_cat'.format(i)] = df[i].astype('category').copy()
    df_cat['{}_cat'.format(i)] = df_cat['{}_cat'.format(i)].cat.codes
    
print(df_cat.head())

X = np.array(df_cat.drop('class_cat', axis = 1))
y = np.array(df_cat['class_cat'])

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


kf = KFold(n_splits=5, random_state = 42)
results = []

for train_index, test_index in kf.split(X):
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     model = RandomForestClassifier(n_estimators = 100, random_state = 24)
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     results.append(f1_score(y_test, y_pred))
         
print("f1-score: ", np.mean(results))
