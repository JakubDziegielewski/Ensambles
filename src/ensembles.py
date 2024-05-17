from sklearn import svm
from sklearn import tree
from sklearn import naive_bayes
from sklearn.base import BaseEstimator
import numpy as np
from collections import Counter
from copy import deepcopy

class SingleClassifier:
    def __init__(self, BaseEstimator):
        self.classifier = BaseEstimator
        self.column_indices = None

class Ensemble:     
    def __init__(self):
        self.ensambles = []
        self.classifier = None
    def fit(self, data_x, data_y, classifiers_number, attributes_subset_number):
        if attributes_subset_number > data_x.shape[1] or attributes_subset_number < 1:
            raise ValueError("Attributes subset number must be between 1 and the number of all attributes in data")
        if classifiers_number < 1:
            raise ValueError("Classifiers number must be positive")
        self.ensambles = []
        for i in range(classifiers_number):
            row_indices = np.random.randint(data_x.shape[0], size = data_x.shape[0])
            single_classifier = SingleClassifier(deepcopy(self.classifier))
            single_classifier.column_indices = np.random.choice(data_x.shape[1], attributes_subset_number, replace=False)
            single_classifier.classifier.fit(data_x[row_indices][:, single_classifier.column_indices], data_y[row_indices])
            self.ensambles.append(single_classifier)
        
    
    def predict(self, data_x):
        prediction = np.zeros(shape=(0, data_x.shape[0]), dtype='int32')
        result = np.array([], dtype='int32')
        for single_classfier in self.ensambles:
            prediction = np.vstack([prediction, single_classfier.classifier.predict(data_x[:, single_classfier.column_indices])])
        for column in prediction.T:
            counter = Counter(column)
            result = np.append(result, counter.most_common(1)[0][0])
        return result
            
            

class SVMEnsemble(Ensemble):
    def __init__(self):
        super().__init__()
        self.classifier = svm.SVC()

class DecisionTreeEnsemble(Ensemble):
    def __init__(self):
        super().__init__()
        self.classifier = tree.DecisionTreeClassifier()

class GaussianNaiveBayesEnseble(Ensemble):
    pass

class DiscreteNaiveBayesEnsemble(Ensemble):
    pass