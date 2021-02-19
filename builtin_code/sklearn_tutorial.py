from sklearn.datasets import load_iris, load_boston, load_digits
import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer


from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



def get_data(dataset):
    df =  None
    try:
        data = dataset
        df = pd.DataFrame(data.data,columns=data.feature_names)
        df['target'] = pd.Series(data.target)
        print(df.head())
    except(AttributeError):
        df = dataset
        plt.gray() 
        for i in range(0,5):
            plt.matshow(data.images[i]) 
            plt.show() 
    return df 
    
get_data(load_iris())
get_data(load_boston())
get_data(load_digits())


df = pd.read_csv('household_power_consumption.txt', sep=';')
df_nan = df[df['Sub_metering_2'] == '?']
df1 = df_nan.sample(5000)
df2 = df.sample(35000)

df = df1.append(df2)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('hpc.csv')

df = pd.read_csv('hpc.csv')
del df['Unnamed: 0']



df['Global_active_power'].replace('?', np.nan, inplace = True)
df['Global_reactive_power'].replace('?', np.nan, inplace = True)
df['Voltage'].replace('?', np.nan, inplace = True)
df['Global_intensity'].replace('?', np.nan, inplace = True)
df['Sub_metering_1'].replace('?', np.nan, inplace = True)
df['Sub_metering_2'].replace('?', np.nan, inplace = True)
print(df.head())
print(df.info())


imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

X = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1','Sub_metering_2' ]]
imp_mean.fit(X)


df_new = pd.DataFrame(imp_mean.transform(X), columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                      'Global_intensity', 'Sub_metering_1','Sub_metering_2' ])
    
df_new['Date'] = df['Date']
df_new['Time'] = df['Time']

print(df_new.head())

print(df_new['Global_intensity'].max())


scaler = StandardScaler()
scaler.fit(np.array(df_new[['Global_intensity']]))
df_new['Global_intensity'] = scaler.transform(np.array(df_new[['Global_intensity']]))


print(df_new.head())
print("Max: ", df_new['Global_intensity'].max())
print("Min: ", df_new['Global_intensity'].min())


normalizer = Normalizer()
normalizer.fit(np.array(df_new[['Sub_metering_2']]))
df_new['Sub_metering_2'] = normalizer.transform(np.array(df_new[['Sub_metering_2']]))

print(df_new.head())
print("Max: ", df_new['Sub_metering_2'].max())
print("Min: ", df_new['Sub_metering_2'].min())


df_housing = get_data(load_boston())

print("Correlation: ", df_housing['AGE'].corr(df_housing['target']))



X = df_housing[['AGE', 'PTRATIO', 'RM']]
y = df_housing[['target']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
rms = mean_squared_error(y_test, y_pred)
r2 =  r2_score(y_test, y_pred)

print("MSE:", rms)
print("R^2:", r2)


from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators= 100, max_depth=10, random_state =42)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
rms = mean_squared_error(y_test, y_pred)
r2 =  r2_score(y_test, y_pred)

print("RF MSE:", rms)
print("RF R^2:", r2)




df_iris= get_data(load_iris())

X = df_iris[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = df_iris[['target']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)


logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

figsize = (10,7)
fontsize=20
conmat =  confusion_matrix(y_test, y_pred)  
val = np.mat(conmat)  
classnames = list(set(df_iris['target']))
df_cm = pd.DataFrame(
    val, index=classnames, columns=classnames, 
)
 
df_cm = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]    
plt.figure(figsize=figsize)
heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues",)
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
plt.ylabel('True label', fontsize=fontsize)
plt.xlabel('Predicted label', fontsize=fontsize)
plt.title('Classification of Iris Plants', fontsize=fontsize)
plt.show()    





from sklearn.ensemble import RandomForestClassifier

rf_cass = RandomForestClassifier(n_estimators= 100, max_depth=10, random_state =42)
rf_cass.fit(X_train, y_train)
y_pred = rf_cass.predict(X_test)



figsize = (10,7)
fontsize=20
conmat =  confusion_matrix(y_test, y_pred)  
val = np.mat(conmat)  
classnames = list(set(df_iris['target']))
df_cm = pd.DataFrame(
    val, index=classnames, columns=classnames, 
)
 
df_cm = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]    
plt.figure(figsize=figsize)
heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues",)
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
plt.ylabel('True label', fontsize=fontsize)
plt.xlabel('Predicted label', fontsize=fontsize)
plt.title('RF Classification of Iris Plants', fontsize=fontsize)
plt.show()   
