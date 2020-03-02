import pandas as pd

df = pd.read_csv("forestfires.csv")

print(df.head())

df['month_cat'] = df['month'].astype('category')
df['month_cat'] = df['month_cat'].cat.codes

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

X = np.array(df[['month_cat', 'temp', 'wind', 'rain']])
y = np.array(df[['area']]).ravel()
result = []
for i in range(0, 1000):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    reg = RandomForestRegressor(n_estimators = 100, max_depth = 100)
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)
    result.append(mean_absolute_error(y_test, y_pred))
    
print("Accuracy: ", np.mean(result))

import os
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle                    
import cv2                                 
import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten


class_names_label = {'mountain': 0,
                    'street' : 1,
                    'glacier' : 2,
                    'buildings' : 3,
                    'sea' : 4,
                    'forest' : 5
                    }

def load_data():
    
    datasets = ['seg_train/seg_train', 'seg_test/seg_test']
    result = []
    
    for dataset in datasets:
        
        images = []
        labels = []
        
        print("Loading {}".format(dataset))
        
        for folder in os.listdir(dataset):
            curr_label = class_names_label[folder]            
            for file in os.listdir(os.path.join(dataset, folder)):
                
                img_path = os.path.join(os.path.join(dataset, folder), file)

                current_image = cv2.imread(image_path)
                current_image = cv2.resize(current_image, (150, 150)) 
            
                images.append(current_image)
                labels.append(current_label)
                
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   
        
        result.append((images, labels))

    return result

(X_train, y_train), (X_test, y_test) = load_data()



model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)))
model.add(MaxPooling2D(2,2))        
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(2,2)) 
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(6, activation = 'softmax'))

        
