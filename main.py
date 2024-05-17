from src.ensembles import DecisionTreeEnsemble
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import tree


iris = load_breast_cancer()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=123
)


clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
a = clf.predict(x_test)

clf_ens = DecisionTreeEnsemble()
clf_ens.fit(x_train, y_train, 50, 5)
b = clf_ens.predict(x_test)
print(sum(b == y_test) / len(y_test))
print(sum(a == y_test) / len(y_test))

