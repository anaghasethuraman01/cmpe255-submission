import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from tabulate import tabulate

        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        print(self.pima.head())
        self.X_test = None
        self.y_test = None
        

    def define_feature(self, feature_cols):
        X = self.pima[feature_cols]
        y = self.pima.label
        return X, y
    
    def train(self,feature_cols,tol):
        # split X and y into training and testing sets
        X, y = self.define_feature(feature_cols)
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, random_state=0)
        # train a logistic regression model on the training set
        logreg = LogisticRegression(tol=tol)
        logreg.fit(X_train, y_train)
        return logreg
    
    def predict(self, feature_cols,tol):
        model = self.train(feature_cols,tol)
        y_pred_class = model.predict(self.X_test)
        return y_pred_class


    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()
    
    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)
    
if __name__ == "__main__":
    classifer = DiabetesClassifier()
    # Baseline
    feature_cols = ['pregnant', 'insulin', 'bmi', 'age']

    result = classifer.predict(feature_cols,0.0001)
        #print(f"Predicition={result}")
    score = classifer.calculate_accuracy(result)
    print(f"score={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"confusion_matrix=${con_matrix}")
    list1=[['Baseline',score,con_matrix,'Baseline features']]
    # Solution 1
    feature_cols = ['pregnant','glucose', 'insulin', 'bmi', 'skin']
    result = classifer.predict(feature_cols,0.0001)
        #print(f"Predicition={result}")
    score = classifer.calculate_accuracy(result)
    print(f"score={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"confusion_matrix=${con_matrix}")
    list2=['Solution 1',score,con_matrix,'using features : pregnant,glucose,insulin,bmi,skin']
    list1.append(list2)

    # Solution 2
    feature_cols = ['pregnant','glucose', 'insulin', 'bmi', 'age']
    result = classifer.predict(feature_cols,0.0001)
        #print(f"Predicition={result}")
    score = classifer.calculate_accuracy(result)
    print(f"score={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"confusion_matrix=${con_matrix}")
    list2=['Solution 2',score,con_matrix,'using features : pregnant,glucose,insulin,bmi,age']
    list1.append(list2)

    
    # Solution 3
    feature_cols = ['pregnant','glucose','insulin', 'bmi', 'age','bp','skin']        
    result = classifer.predict(feature_cols,10)
    #print(f"Predicition={result}")
    score = classifer.calculate_accuracy(result)
    print(f"score={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"confusion_matrix=${con_matrix}")
    list2=['Solution 3',score,con_matrix,'using features : pregnant,glucose,insulin,bmi,age,bp,skin and tol=10']
    list1.append(list2)

    df=np.array(list1)
    print('___________________________________________________________')
    print("Expected Output")
    print('___________________________________________________________')
    print(tabulate( df,headers=['Experiment', 'Accuracy', 'Confusion Matrix', 'Comments']))
    print('___________________________________________________________')t
