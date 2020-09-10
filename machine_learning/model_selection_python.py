#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:02:51 2020

@author: sadrachpierre
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

df = pd.read_csv("winemag-data-130k-v2.csv")
print(len(df))

del df['Unnamed: 0']

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(df.head())

df['country_cat'] = df['country'].astype('category').copy()
df['country_cat'] = df['country_cat'].cat.codes

df['winery_cat'] = df['winery'].astype('category').copy()
df['winery_cat'] = df['winery_cat'].cat.codes

df['variety_cat'] = df['variety'].astype('category').copy()
df['variety_cat'] = df['variety_cat'].cat.codes
df.fillna(0, inplace=True)
   
df['price_class']=np.where(df['price']>=50,1,0)

X = np.array(df[['country_cat', 'winery_cat', 'variety_cat', 'points']])
y = np.array(df['price_class'])

X_train, X_test, y_train, y_test = train_test_split(X,y)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]


fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

sns.set()
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Wine Price Classification')
plt.legend(loc="lower right")
plt.show()


precision, recall, _ = precision_recall_curve(y_test,y_pred)
average_precision = average_precision_score(y_test, y_pred)
                                                    

sns.set()
plt.figure()
plt.step(recall, precision, where='post')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score Wine Price Classification, averaged over all classes: AP={0:0.2f}'
    .format(average_precision))


folds = KFold()
folds.get_n_splits(df)
y_true  = []
y_pred = []
for train_index, test_index in folds.split(df):
    df_test = df.iloc[test_index]
    df_train = df.iloc[train_index]     
    X_train = np.array(df_train[['country_cat', 'winery_cat', 'variety_cat', 'points']])
    y_train = np.array(df_train['price_class'])
    X_test = np.array(df_test[['country_cat', 'winery_cat', 'variety_cat', 'points']])
    y_test = np.array(df_test['price_class'])    
    y_true.append(y_test)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)    
    y_pred.append(model.predict_proba(X_test)[:,1])
y_pred = [item for sublist in y_pred for item in sublist]   
y_true = [item for sublist in y_true for item in sublist] 

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

sns.set()
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Wine Price Classification (KFolds)')
plt.legend(loc="lower right")
plt.show()


precision, recall, _ = precision_recall_curve(y_true,y_pred)
average_precision = average_precision_score(y_true, y_pred)
                                                    

sns.set()
plt.figure()
plt.step(recall, precision, where='post')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score Wine Price Classification (KFolds), averaged over all classes: AP={0:0.2f}'
    .format(average_precision))





df = df.sample(n=5000)
loo = LeaveOneOut()
loo.get_n_splits(df)
y_true  = []
y_pred = []
for train_index, test_index in loo.split(df):
    df_test = df.iloc[test_index]
    df_train = df.iloc[train_index]     
    X_train = np.array(df_train[['country_cat', 'winery_cat', 'variety_cat', 'points']])
    y_train = np.array(df_train['price_class'])
    X_test = np.array(df_test[['country_cat', 'winery_cat', 'variety_cat', 'points']])
    y_test = np.array(df_test['price_class'])    
    y_true.append(y_test)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)    
    y_pred.append(model.predict_proba(X_test)[:,1])
y_pred = [item for sublist in y_pred for item in sublist]   
y_true = [item for sublist in y_true for item in sublist] 

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

sns.set()
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Wine Price Classification (LOOCV)')
plt.legend(loc="lower right")
plt.show()


precision, recall, _ = precision_recall_curve(y_true,y_pred)
average_precision = average_precision_score(y_true, y_pred)
                                                    

sns.set()
plt.figure()
plt.step(recall, precision, where='post')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score Wine Price Classification (LOOCV), averaged over all classes: AP={0:0.2f}'
    .format(average_precision))