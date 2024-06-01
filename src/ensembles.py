from sklearn.naive_bayes import CategoricalNB
import numpy as np
from collections import Counter
from typing import Literal, Callable
from numpy.random import RandomState
from numpy.typing import ArrayLike



class SingleClassifier:
    def __init__(self, classifier):
        self.classifier = classifier
        self.column_indices = None


class Ensemble:
    def __init__(
        self,
        classifier_constructor: Callable,
        classifiers_number: int = 100,
        max_attributes: float | int | Literal["sqrt", "log2"] = "sqrt",
        random_state: int | None = None,
        min_categories: ArrayLike | int | None = None
    ):
        if classifiers_number < 1:
            raise ValueError("Classifiers number must be positive")
        self.classifier_constructor = classifier_constructor
        self.ensambles = []
        self.classifier = None
        self.classifiers_number = classifiers_number
        self.max_attributes = max_attributes
        self.random_state = RandomState(random_state)
        self.min_categories = min_categories

    def fit(self, data_x, data_y):
        all_attributes = data_x.shape[1]
        max_attributes = None
        if type(self.max_attributes) is int:
            if self.max_attributes < 1 or self.max_attributes > all_attributes:
                raise ValueError(
                    f"Attributes subset number must be between 1 and the number of all attributes in data ({all_attributes})"
                )
            max_attributes = self.max_attributes
        elif type(self.max_attributes) is float:
            if self.max_attributes > 1 or self.max_attributes < 0:
                raise ValueError(
                    "Wrong fraction of attributes: cannot be smaller than 0 and bigger than 1"
                )
            max_attributes = int(np.round(all_attributes * self.max_attributes))
        elif self.max_attributes == "sqrt":
            max_attributes = int(np.sqrt(all_attributes))
        elif self.max_attributes == "log":
            max_attributes = int(np.log(all_attributes))

        self.ensambles = []
        for _ in range(self.classifiers_number):
            row_indices = self.random_state.choice(data_x.shape[0], size=data_x.shape[0])    
            column_indices = self.random_state.choice(
                data_x.shape[1], max_attributes, replace=False
            )
            single_classifier = self.classifier_constructor()
            if type(single_classifier) is CategoricalNB:
                if self.min_categories is None:
                    raise ValueError("Provide min_categories attribute for CategoricalBN classifiers")
                single_classifier = self.classifier_constructor(min_categories = np.array(self.min_categories)[column_indices])
            
            single_classifier.fit(
                data_x[row_indices][:, column_indices],
                data_y[row_indices]
            )
            self.ensambles.append((single_classifier, column_indices))

    def predict(self, data_x):
        prediction = np.zeros(shape=(0, data_x.shape[0]), dtype="int32")
        result = np.array([], dtype="int32")
        for single_classfier, column_indices in self.ensambles:
            prediction = np.vstack(
                [
                    prediction,
                    single_classfier.predict(
                        data_x[:, column_indices]
                    )
                ]
            )
        for column in prediction.T:
            counter = Counter(column)
            result = np.append(result, counter.most_common(1)[0][0])
        return result
    
    def score(self, data_x, data_y):
        predictions = self.predict(data_x)
        return sum(predictions == data_y) / len(data_y)

