#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:55:45 2020

@author: sadrachpierre
"""

import pandas as pd 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("Customer-Churn.csv")
print(df.head())
df.gender = pd.Categorical(df.gender)
df['gender_code'] = df.gender.cat.codes

df.SeniorCitizen = pd.Categorical(df.SeniorCitizen)
df['SeniorCitizen_code'] = df.SeniorCitizen.cat.codes


df.PhoneService = pd.Categorical(df.PhoneService)
df['PhoneService_code'] = df.PhoneService.cat.codes

df.MultipleLines = pd.Categorical(df.MultipleLines)
df['MultipleLines_code'] = df.MultipleLines.cat.codes

df.InternetService = pd.Categorical(df.InternetService)
df['InternetService_code'] = df.InternetService.cat.codes


df.Partner = pd.Categorical(df.Partner)
df['Partner_code'] = df.Partner.cat.codes

df.Dependents = pd.Categorical(df.Dependents)
df['Dependents_code'] = df.Dependents.cat.codes

df.OnlineSecurity = pd.Categorical(df.OnlineSecurity)
df['OnlineSecurity_code'] = df.OnlineSecurity.cat.codes

df.OnlineBackup = pd.Categorical(df.OnlineBackup)
df['OnlineBackup_code'] = df.OnlineBackup.cat.codes

df.DeviceProtection = pd.Categorical(df.DeviceProtection)
df['DeviceProtection_code'] = df.DeviceProtection.cat.codes


df.TechSupport = pd.Categorical(df.TechSupport)
df['TechSupport_code'] = df.TechSupport.cat.codes

df.StreamingTV = pd.Categorical(df.StreamingTV)
df['StreamingTV_code'] = df.StreamingTV.cat.codes


df.StreamingMovies = pd.Categorical(df.StreamingMovies)
df['StreamingMovies_code'] = df.StreamingMovies.cat.codes

df.Contract = pd.Categorical(df.Contract)
df['Contract_code'] = df.Contract.cat.codes

df.PaperlessBilling = pd.Categorical(df.PaperlessBilling)
df['PaperlessBilling_code'] = df.PaperlessBilling.cat.codes

df.PaymentMethod = pd.Categorical(df.PaymentMethod)
df['PaymentMethod_code'] = df.PaymentMethod.cat.codes

df.Churn = pd.Categorical(df.Churn)
df['Churn_code'] = df.Churn.cat.codes

df['tenure'] = df['tenure'].astype(float)
df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
#df['TotalCharges'] = df['TotalCharges'].astype(float)

import numpy as np

features = ['gender_code', 'SeniorCitizen_code', 'PhoneService_code', 'MultipleLines_code', 
                 'InternetService_code', 'Partner_code', 'Dependents_code', 'PaymentMethod_code', 
                 'PaymentMethod_code', 'PaperlessBilling_code','Contract_code', 'StreamingMovies_code',
                 'StreamingTV_code', 'TechSupport_code', 'DeviceProtection_code', 'OnlineBackup_code',
                 'OnlineSecurity_code', 'Dependents_code', 'Partner_code','tenure', 'MonthlyCharges']

X = np.array(df[features])
y = np.array(df['Churn_code'])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = Sequential()
model.add(Dense(8, input_shape = (len(features),)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs = 1)

y_pred = [round(float(x)) for x in model.predict(X_test)]

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
print("roc_auc_score: ", roc_auc_score(y_test, y_pred))


from sklearn.metrics import f1_score
print("f1 score: ", f1_score(y_test, y_pred))
