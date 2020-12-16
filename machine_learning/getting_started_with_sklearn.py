from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np 

from collections import Counter

df = pd.read_csv("Bank_churn_modelling.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df.head())
class_names = list(set(df['Exited']))

df_train = df.sample(int(0.67*len(df)), random_state = 42)
df_test = df[~df['CustomerId'].isin(list(df_train['CustomerId']))]


sample_in = int(min(list(dict(Counter(df_train['Exited'])).values()))-1)
df_1 = df_train[df_train['Exited'] == 0]
df_2 = df_train[df_train['Exited'] == 1]
df_1 =df_1.sample(n=sample_in, random_state = 42)
df_2 =df_2.sample(n=sample_in, random_state = 24)
df_train = df_1.append(df_2)


X_train = df_train[['CreditScore', 'Age', 'Tenure', 'NumOfProducts']]
y_train = df_train['Exited']
X_test =  df_test[['CreditScore', 'Age', 'Tenure', 'NumOfProducts']]
y_test = df_test['Exited']



def get_rf_parameters():
    n_estimators = [10, 50, 100]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [5, 10, 20, 50, None]
    min_samples_split = [2, 4, 6, 8]
    min_samples_leaf = [1, 2, 4, 6]
    bootstrap = [True, False]
    
    random_grid = {'n_estimators':n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap':bootstrap}
    model = RandomForestClassifier(random_state =42)
    rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, 
                             n_iter = 5, cv = 3, verbose =  2, random_state = 42)
    rf_random.fit(X_train, y_train)
    parameters = rf_random.best_params_
    
    return parameters

#rf_par = get_rf_parameters()



def get_svm_parameters():
    C = [0.1, 1, 10,]
    gamma = [1, 0.1, 0.01]
    kernel = ['rbf', 'linear']
    random_grid = {'C': C,'gamma': gamma, 'kernel':kernel}  
    model = SVC(random_state =42)
    svm_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, 
                             n_iter = 5, cv = 3, verbose =  2, random_state = 42)
    svm_random.fit(X_train, y_train)
    parameters = svm_random.best_params_
    
    return parameters
    
svm_par = get_svm_parameters()


def train_model(model_type, rf_parameters, svm_parameters):
    
    if rf_parameters:
        model = {'logistic_regression': LogisticRegression(), 'random_forests': RandomForestClassifier(**rf_parameters, random_state =42), 
                 'SVM': SVC()}
    elif svm_parameters:
        model = {'logistic_regression': LogisticRegression(), 'random_forests': RandomForestClassifier(random_state =42), 
                 'SVM': SVC(**svm_parameters)}

    model = model[model_type]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    conmat = confusion_matrix(y_test, y_pred)
    conmat = np.mat(conmat)
    return y_pred, conmat

#y_pred, conmat = train_model('logistic_regression')
#y_pred, conmat = train_model('random_forests', rf_par)
y_pred, conmat = train_model('SVM', None, svm_par)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xlim(-0.5, len(np.unique(y))-0.5)
    plt.ylim(len(np.unique(y))-0.5, -0.5)
    return ax


np.set_printoptions(precision=2)


# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
