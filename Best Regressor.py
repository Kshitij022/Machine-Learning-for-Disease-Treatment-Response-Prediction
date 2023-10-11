import json

import numpy as np
import pandas as pd
from joblib import load
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

#please add your data into read_excel() on line 14 and adjust your file locations as needed

class Regressor_Test:
    def __init__(self):
        model = load('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/COMP3009 Machine Learning/Submissions/Assignment 2/regressor.joblib') #random forest regressor
        retained_features = json.load(open('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/COMP3009 Machine Learning/Submissions/Assignment 2/retained_features.json'))
        X, id = self.preprocess(retained_features, pd.read_excel('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/COMP3009 Machine Learning/Submissions/Assignment 2/testDatasetExample.xls')) #add data here
        self.test(model, X, id)

    def preprocess(self, features, data):
        id = data.iloc[:, 0]
        X = data[features]
        
        #adjust 999s
        col_list = data.columns
        for i in range(3, len(col_list)) :
            data[col_list[i]].replace(to_replace = 999, value = data[col_list[i]].mean(), inplace = True)

        #minmax normalisation
        #scaler = MinMaxScaler()
        scaler = load('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/COMP3009 Machine Learning/Submissions/Assignment 2/scaler.joblib')
        X_norm = scaler.transform(X)

        return X_norm, id

    def test(self, model, X, id):
        output = pd.DataFrame(model.predict(X), columns = ['predictions'])
        id = pd.DataFrame(id, columns = ['ID'])
        concat = id.join(output)
        concat.to_csv('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/COMP3009 Machine Learning/Submissions/Assignment 2/regressor_predictions.csv', index = False)

def main():
    Regressor_Test()

if __name__ == '__main__':
    main()