from src.ensembles import SVMEnsemble
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=123
)

clf = svm.SVC()
clf.fit(x_train, y_train)
print(clf.predict(x_test))

clf_ens = SVMEnsemble()
clf_ens.fit(x_train, y_train, 1, 4)
print(clf_ens.predict(x_test))
