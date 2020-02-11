from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

import pandas as pd

class Model:
    def __init__(self, datafile = "weatherHistory.csv"):
        self.df = pd.read_csv(datafile)
        self.linear_reg = LinearRegression()
        self.random_forest = RandomForestRegressor()
    def split(self, test_size):
        X = np.array(self.df[['Humidity', 'Pressure (millibars)']])
        y = np.array(self.df['Temperature (C)'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    
    def fit(self):
        self.model = self.random_forest.fit(self.X_train, self.y_train)
    
    def predict(self, input_value):
        if input_value == None:
            result = self.random_forest.predict(self.X_test)
        else:
            result = self.random_forest.predict(np.array(input_value))
        return result
if __name__ == '__main__':
    model_instance = Model()
    model_instance.split(0.2)
    model_instance.fit()    
    print("Accuracy: ", model_instance.model.score(model_instance.X_test, model_instance.y_test))


        
        
     
