import json

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import dump
from keras import layers, models
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class Preprocess:
    def __init__(self, dataset):
        data = self.data_cleaning_remove(dataset)
        clean_data = self.data_cleaning_replace_median(data) #alternate between data_cleaning functions 
        X, Y_PCR, Y_RFS = self.data_split(clean_data)
        X = self.feature_selection(X) #alternate between feature_selection and dim_reduction functions
        Y_PCR = Y_PCR.iloc[:, 0].values
        Y_RFS = Y_RFS.iloc[:, 0].values
        X_norm = self.normalisation_minmax(X) #alternate between normalisation functions
        self.X_train_PCR, self.X_test_PCR, self.Y_train_PCR, self.Y_test_PCR = train_test_split(X_norm, Y_PCR, test_size = 0.2, random_state = 42)
        self.X_train_RFS, self.X_test_RFS, self.Y_train_RFS, self.Y_test_RFS = train_test_split(X_norm, Y_RFS, test_size = 0.2, random_state = 42)
        
    def data_cleaning_remove(self, data):
        data['pCR (outcome)'] = data['pCR (outcome)'].replace(999, np.nan)
        data['RelapseFreeSurvival (outcome)'] = data['RelapseFreeSurvival (outcome)'].replace(999, np.nan)
        data = data.dropna()

        return data

    def data_cleaning_replace_mean(self, data):
        col_list = data.columns
        for i in range(3, len(col_list)):
            data[col_list[i]].replace(to_replace = 999, value = data[col_list[i]].median(), inplace = True)
            data['Age'] = round(data['Age'] / 20, 0)

        return data

    def data_cleaning_replace_median(self, data):
        col_list = data.columns
        for i in range(3, len(col_list)) :
            data[col_list[i]].replace(to_replace = 999, value = data[col_list[i]].mean(), inplace = True)
            data['Age'] = data['Age'] // 10

        return data

    def data_split(self, data):
        X = data.copy(deep = True)
        X.drop(columns = data.columns[:3], axis = 1, inplace = True)

        Y_PCR = data[['pCR (outcome)']].copy(deep = True)
        Y_RFS = data[['RelapseFreeSurvival (outcome)']].copy(deep = True)

        return X, Y_PCR, Y_RFS

    def feature_selection(self, X):
        data_copy = X.columns
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        X.drop(columns = to_drop, axis = 1,inplace = True)

        selector = VarianceThreshold(threshold = (0.95 * (1 - 0.95)))
        X = selector.fit_transform(X)
        features = selector.get_support(indices = True)
        features = [column for column in data_copy[features]]
        json.dump(features, open('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/COMP3009 Machine Learning/Submissions/Assignment 2/retained_features.json', 'w'))
        
        return X

    def dim_reduction(self, X):
        pca = PCA(n_components = 33)
        X_PCA = pca.fit_transform(X)

        return X_PCA

    def normalisation_zmean(self, X):
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)

        return X_norm

    def normalisation_minmax(self, X):
        scaler = RobustScaler()
        scaler.fit(X)
        X_norm = scaler.transform(X)
        dump(scaler, open('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/COMP3009 Machine Learning/Submissions/Assignment 2/scaler.joblib', 'wb'))

        return X_norm

#classification

class Classification_NeuralNetwork:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)
        
        folds = 5
        activation = 'relu'
        kf = StratifiedKFold(folds, shuffle = True, random_state = 42) 
        fold = 0
        accuracy_per_fold = []

        for(train, test) in kf.split(inputs, targets):
            fold += 1
            print('Fold: {}'.format(fold))

            model = models.Sequential([
                                       layers.Dense(128, input_shape = (49, ), activation = activation),
                                       layers.Dropout(0.1),
                                       layers.Dense(128, activation = activation),
                                       layers.Dropout(0.1),
                                       layers.Dense(256, activation = activation),
                                       layers.Dropout(0.1),
                                       layers.Dense(64, activation = activation),
                                       layers.Dense(1, activation = 'sigmoid')
            ])

            model.compile(
                          loss = 'binary_crossentropy', 
                          optimizer = 'adam', 
                          metrics = ['accuracy']
            )

            history = model.fit(inputs[train], targets[train], epochs = 2000)
            validation_score = model.evaluate(inputs[test], targets[test], verbose = 0)
            accuracy_per_fold.append(validation_score[1] * 100)

        return 'Training Accuracy: {}%, {}-Fold Cross-Validation Accuracy: {}%'.format(np.round(history.history['accuracy'][-1] * 100, 2), folds, np.round(sum(accuracy_per_fold) / folds), 2)

class Classification_XGBoost:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)

        model = xgb.XGBClassifier(objective = 'binary:logistic', random_state = 42)
        scores = cross_validate(model, inputs, targets, cv = 5, scoring = 'accuracy', return_train_score = True)
        
        return 'Training Accuracy: {}%, 5-Fold Cross-Validation Accuracy: {}%'.format(round(np.mean(scores['train_score']) * 100, 2), round(np.mean(scores['test_score']) * 100, 2))

class Classification_SVM:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)
        
        model = SVC(
        kernel = 'poly',
        C = 1,
        degree = 5,
        coef0 = 0.001,
        gamma = 'auto'
        )

        scores = cross_validate(model, inputs, targets, cv = 5, scoring = 'accuracy', return_train_score = True)
        predictions = model.fit(inputs, targets)
        dump(model, open('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/COMP3009 Machine Learning/Submissions/Assignment 2/classifier.joblib', 'wb')) #MinMax, replace NaN with median

        #predictions = model.predict(input) --------------------------------> how to use on unseen data??
        #fpr, tpr, thresholds = metrics.auc_curve(targets, predictions)
        #roc_auc = metrics.auc(fpr, tpr)
        #display = metrics.RocCurveDisplay(fpr, tpr, roc_auc)
        #display.plot()
        #plt.show()

        #conf_matrix = metrics.confusion_matrix(targets, predictions, labels = model.classes_) ----------------------> perhaps needs to go in testing files??
        #display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = model.classes_)
        #display.plot()
        #plt.show()

        return 'Training Accuracy: {}%, 5-Fold Cross-Validation Accuracy: {}%'.format(round(np.mean(scores['train_score']) * 100, 2), round(np.mean(scores['test_score']) * 100, 2))

class Classification_LogisticRegression:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)
        
        model = LogisticRegression(
        penalty = 'l2',
        random_state = 42,
        solver = 'saga',
        max_iter = 100
        )

        scores = cross_validate(model, inputs, targets, cv = 5, scoring = 'accuracy', return_train_score = True)
        
        return 'Training Accuracy: {}%, 5-Fold Cross-Validation Accuracy: {}%'.format(round(np.mean(scores['train_score']) * 100, 2), round(np.mean(scores['test_score']) * 100, 2))

class Classification_KNN:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)
        
        model = KNeighborsClassifier(n_neighbors = 5)
        scores = cross_validate(model, inputs, targets, cv = 5, scoring = 'accuracy', return_train_score = True)
        
        return 'Training Accuracy: {}%, 5-Fold Cross-Validation Accuracy: {}%'.format(round(np.mean(scores['train_score']) * 100, 2), round(np.mean(scores['test_score']) * 100, 2))

class Classification_DecisionTree:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)
        
        model = DecisionTreeClassifier(criterion = 'gini', splitter = 'best', max_leaf_nodes = 4, random_state = 42)
        scores = cross_validate(model, inputs, targets, cv = 5, scoring = 'accuracy', return_train_score = True)
        
        return 'Training Accuracy: {}%, 5-Fold Cross-Validation Accuracy: {}%'.format(round(np.mean(scores['train_score']) * 100, 2), round(np.mean(scores['test_score']) * 100, 2))

class Classification_RandomForest:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)
        
        model = RandomForestClassifier(criterion = 'gini', max_leaf_nodes = 4, random_state = 42)
        scores = cross_validate(model, inputs, targets, cv = 5, scoring = 'accuracy', return_train_score = True)
        
        return 'Training Accuracy: {}%, 5-Fold Cross-Validation Accuracy: {}%'.format(round(np.mean(scores['train_score']) * 100, 2), round(np.mean(scores['test_score']) * 100, 2))

class Run_Classification:
    def __init__(self):
        dataset = pd.read_excel('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/COMP3009 Machine Learning/Submissions/Assignment 2/trainDataset.xls')
        data = Preprocess(dataset)
        #ann = Classification_NeuralNetwork(data.X_train_PCR, data.X_test_PCR, data.Y_train_PCR, data.Y_test_PCR)
        xgb = Classification_XGBoost(data.X_train_PCR, data.X_test_PCR, data.Y_train_PCR, data.Y_test_PCR)
        svm = Classification_SVM(data.X_train_PCR, data.X_test_PCR, data.Y_train_PCR, data.Y_test_PCR)
        lr = Classification_LogisticRegression(data.X_train_PCR, data.X_test_PCR, data.Y_train_PCR, data.Y_test_PCR)
        knn = Classification_KNN(data.X_train_PCR, data.X_test_PCR, data.Y_train_PCR, data.Y_test_PCR)
        dt = Classification_DecisionTree(data.X_train_PCR, data.X_test_PCR, data.Y_train_PCR, data.Y_test_PCR)
        rf = Classification_RandomForest(data.X_train_PCR, data.X_test_PCR, data.Y_train_PCR, data.Y_test_PCR)

        #print('\nClassification: Neural Network ', ann.accuracy)
        print('\nClassification: XGBoost ', xgb.accuracy)
        print('Classification: SVC ', svm.accuracy)
        print('Classification: Logistic Regression ', lr.accuracy)
        print('Classification: KNN ', knn.accuracy)
        print('Classification: Decision Tree ', dt.accuracy)
        print('Classification: Random Forest ', rf.accuracy, '\n')

#regression

class Regression_NeuralNetwork:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)
        
        folds = 5
        activation = 'relu'
        kf = KFold(folds, shuffle = True, random_state = 42) 
        fold = 0
        loss_per_fold = []

        for(train, test) in kf.split(inputs, targets):
            fold += 1
            print('Fold: {}'.format(fold))

            model = models.Sequential([
                                       layers.Dense(128, input_shape = (49, ), activation = activation),
                                       layers.Dropout(0.1),
                                       layers.Dense(128, activation = activation),
                                       layers.Dropout(0.1),
                                       layers.Dense(256, activation = activation),
                                       layers.Dropout(0.1),
                                       layers.Dense(64, activation = activation),
                                       layers.Dense(1, activation = 'relu')
            ])

            model.compile(
                          loss = 'mean_absolute_error', 
                          optimizer = 'adam'
            )

            history = model.fit(inputs[train], targets[train], epochs = 2000)
            validation_score = model.evaluate(inputs[test], targets[test], verbose = 0)
            loss_per_fold.append(validation_score)

        return 'Training Loss: {}, {}-Fold Cross-Validation Loss: {}'.format(np.round(history.history['loss'][-1], 2), folds, np.round(np.sum(loss_per_fold) / folds, 2))

class Regression_XGBoost:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)

        #grid search used
        model = xgb.XGBRegressor(
            objective = 'reg:squarederror', 
            colsample_bytree = 0.4, 
            eta = 0.1, 
            max_depth = 7, 
            n_estimators = 100,
            seed = 10,
            subsample = 0.7)

        scores = cross_validate(model, inputs, targets, cv = 5, scoring = ['r2', 'neg_mean_absolute_error'], return_train_score = True)
        
        return '5-Fold Cross-Validation Loss: {}, R2 Score: {}'.format(abs(np.mean(scores['test_neg_mean_absolute_error'])), np.mean(scores['test_r2']))

class Regression_SVM:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)

        model = SVR(kernel = 'rbf', C = 1, degree = 3, coef0 = 0.001, gamma = 'scale')
        #param = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                 #'C': np.arange(1, 42, 10),
                 #'degree': np.arange(3, 6),
                 #'coef0': np.arange(0.001, 5, 0.5),
                 #'gamma': ('auto', 'scale')}

        #grids = GridSearchCV(estimator = model, param_grid = param)
        #grids.fit(inputs, targets)
        scores = cross_validate(model, inputs, targets, cv = 5, scoring = ['r2', 'neg_mean_absolute_error'], return_train_score = True)
        
        return '5-Fold Cross-Validation Loss: {}, R2 Score: {}'.format(abs(np.mean(scores['test_neg_mean_absolute_error'])), np.mean(scores['test_r2']))

class Regression_LinearRegression:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)
        
        model = LinearRegression()
        scores = cross_validate(model, inputs, targets, cv = 5, scoring = ['r2', 'neg_mean_absolute_error'], return_train_score = True)
        
        return '5-Fold Cross-Validation Loss: {}, R2 Score: {}'.format(abs(np.mean(scores['test_neg_mean_absolute_error'])), np.mean(scores['test_r2']))

class Regression_KNN:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)

        model = KNeighborsRegressor(n_neighbors = 5)
        scores = cross_validate(model, inputs, targets, cv = 5, scoring = ['r2', 'neg_mean_absolute_error'], return_train_score = True)
        
        return '5-Fold Cross-Validation Loss: {}, R2 Score: {}'.format(abs(np.mean(scores['test_neg_mean_absolute_error'])), np.mean(scores['test_r2']))

class Regression_DecisionTree:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)
        
        model = DecisionTreeRegressor(criterion = 'absolute_error', splitter = 'best', max_leaf_nodes = 4, random_state = 42)
        scores = cross_validate(model, inputs, targets, cv = 5, scoring = ['r2', 'neg_mean_absolute_error'], return_train_score = True)
        
        return '5-Fold Cross-Validation Loss: {}, R2 Score: {}'.format(abs(np.mean(scores['test_neg_mean_absolute_error'])), np.mean(scores['test_r2']))

class Regression_RandomForest:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)
        
        #grid search used
        model = RandomForestRegressor(
            criterion = 'absolute_error', 
            max_depth = 3,
            max_samples = 0.6,
            min_samples_leaf = 1,
            n_estimators = 150,
            random_state = 1 
            )

        scores = cross_validate(model, inputs, targets, cv = 5, scoring = ['r2', 'neg_mean_absolute_error'], return_train_score = True)
        model.fit(inputs, targets)

        dump(model, open('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/COMP3009 Machine Learning/Submissions/Assignment 2/regressor.joblib', 'wb'))
        
        return '5-Fold Cross-Validation Loss: {}, R2 Score: {}'.format(abs(np.mean(scores['test_neg_mean_absolute_error'])), np.mean(scores['test_r2']))

class Run_Regression:
    def __init__(self):
        dataset = pd.read_excel('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/COMP3009 Machine Learning/Submissions/Assignment 2/trainDataset.xls')
        data = Preprocess(dataset)
        #ann = Regression_NeuralNetwork(data.X_train_RFS, data.X_test_RFS, data.Y_train_RFS, data.Y_test_RFS)
        xgb = Regression_XGBoost(data.X_train_RFS, data.X_test_RFS, data.Y_train_RFS, data.Y_test_RFS)
        svm = Regression_SVM(data.X_train_PCR, data.X_test_RFS, data.Y_train_RFS, data.Y_test_RFS)
        lr = Regression_LinearRegression(data.X_train_RFS, data.X_test_RFS, data.Y_train_RFS, data.Y_test_RFS)
        knn = Regression_KNN(data.X_train_RFS, data.X_test_RFS, data.Y_train_RFS, data.Y_test_RFS)
        dt = Regression_DecisionTree(data.X_train_RFS, data.X_test_RFS, data.Y_train_RFS, data.Y_test_RFS)
        rf = Regression_RandomForest(data.X_train_RFS, data.X_test_RFS, data.Y_train_RFS, data.Y_test_RFS)

        #print('\nRegression: Neural Network ', ann.accuracy)
        print('\nRegression: XGBoost ', xgb.accuracy)
        print('Regression: SVR ', svm.accuracy)
        print('Regression: Linear Regression ', lr.accuracy)
        print('Regression: KNN ', knn.accuracy)
        print('Regression: Decision Tree ', dt.accuracy)
        print('Regression: Random Forest ', rf.accuracy, '\n')

def main():
    Run_Classification()
    Run_Regression()

if __name__ == '__main__':
    main()
