import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score

from src.ensembles import Ensemble

def find_intervals(x_train, group_vector):  # auxilary values for data disrcetization
    intervals = np.array([np.zeros(i - 1) for i in group_vector])

    for i, features in enumerate(x_train.T):
        max_value = max(features)
        min_value = min(features)
        section_size = (max_value - min_value) / group_vector[i]
        intervals[i] = np.array(
            [min_value + section_size * j for j in range(1, group_vector[i])]
        )
    return intervals

def _test_classifier(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    
    accuracy_score = clf.score(x_test, y_test)
    y_pred_proba = clf.predict_proba(x_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba, average='macro', multi_class='ovr')
    pr_auc = average_precision_score(y_test, y_pred_proba, average='macro')
    y_pred = clf.predict(x_test)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy_score, roc_auc, pr_auc, precision, recall, conf_matrix

def _test_cv_classifier(clf, x, y, n=10):
   scores = cross_val_score(clf, x, y, cv=n)
   scores_mean = scores.mean()
   return scores, scores_mean

def print_results(accuracy_score, roc_auc, pr_auc, precision, recall, conf_matrix, scores, scores_mean):
    print("Accuracy: {:.3f}".format(accuracy_score))
    print("ROC AUC: {:.3f}".format(roc_auc))
    print("PR AUC: {:.3f}".format(pr_auc))
    print("Precision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("Confusion matrix:\n", conf_matrix)
    
    print("CV scores:", end=" ")
    for score in scores:
        print("{:.3f}".format(score), end=" ")
    print()
    
    print("CV mean score: {:.3f}".format(scores_mean))
    
def _format_conf_matrix(conf_matrix):
    formatted_matrix = '[' + ',<br> '.join(['[' + ', '.join(map(str, row)) + ']' for row in conf_matrix]) + ']'
    return formatted_matrix

# Define a function to format CV scores into multiple lines
def _format_cv_scores(cv_scores):
    lines = []
    for i in range(0, len(cv_scores), 4):
        line = ', '.join(f'{score:.3f}' for score in cv_scores[i:i+3])
        lines.append(line)
    return '<br>'.join(lines)

def test_ensemble( 
    x,
    y,
    x_train,
    y_train,
    x_test,
    y_test,
    classifier,
    max_features_values=[1.0],
    model=None, 
    n_estimators_values=[5,10,50,100],
    random_state=3,
    min_categories=[4] * 561
):
    results = []

    for max_features in max_features_values:
        for n_estimators in n_estimators_values:
            if classifier == RandomForestClassifier:
                clf = classifier(n_estimators=n_estimators, 
                                 max_features=max_features, 
                                 random_state=random_state)
                accuracy_score, roc_auc, pr_auc, precision, recall, conf_matrix = _test_classifier(clf, x_train, y_train, x_test, y_test)
                scores, scores_mean = _test_cv_classifier(clf, x, y, n=10)
                model_name = 'Random Forest'
            elif classifier == Ensemble:
                if model == CategoricalNB:
                    clf = classifier(classifier_constructor=model, 
                                     classifiers_number=n_estimators, 
                                     max_attributes=max_features, 
                                     random_state=random_state, 
                                     min_categories=min_categories)
                else:
                    clf = classifier(classifier_constructor=model, 
                                     classifiers_number=n_estimators, 
                                     max_attributes=max_features,
                                     random_state=random_state)
                accuracy_score, roc_auc, pr_auc, precision, recall, conf_matrix = _test_classifier(clf, x_train, y_train, x_test, y_test)
                scores, scores_mean = _test_cv_classifier(clf, x, y, n=10)
                model_name = model.__class__.__name__ if model is not None else 'Ensemble'
            elif classifier == BaggingClassifier:
                clf = classifier(estimator=model, 
                                 n_estimators=n_estimators, 
                                 max_features=max_features,
                                 random_state=random_state)
                accuracy_score, roc_auc, pr_auc, precision, recall, conf_matrix = _test_classifier(clf, x_train, y_train, x_test, y_test)
                scores, scores_mean = _test_cv_classifier(clf, x, y, n=10)
                model_name = model.__class__.__name__ if model is not None else 'Bagging'
            else:
                print("Unknown classifier")
                continue
            
            results.append({
                'Model': model_name,
                'n_estimators': n_estimators,
                'max_features': max_features,
                'Accuracy': round(accuracy_score, 3),
                'ROC AUC': round(roc_auc, 3),
                'PR AUC': round(pr_auc, 3),
                'Precision': round(precision, 3),
                'Recall': round(recall, 3),
                'Confusion matrix': _format_conf_matrix(conf_matrix),
                'CV scores': _format_cv_scores(scores),
                'CV mean score': round(scores_mean, 3)
            })

    results_df = pd.DataFrame(results)
    return results_df


def test_clf( 
    x,
    y,
    x_train,
    y_train,
    x_test,
    y_test,
    classifier,
    random_state = 3,
    min_categories = [4] * 561
    ):
    
    results = []
    
    if classifier == SVC:
        clf = classifier(probability=True, random_state=random_state)
        accuracy_score, roc_auc, pr_auc, precision, recall,\
        conf_matrix = _test_classifier(clf, x_train, y_train, x_test, y_test)
        scores, scores_mean = _test_cv_classifier(clf, x, y, n=10)
        model = "SVC"
    
    elif classifier == GaussianNB:
        clf = classifier()
        accuracy_score, roc_auc, pr_auc, precision, recall,\
        conf_matrix = _test_classifier(clf, x_train, y_train, x_test, y_test)
        scores, scores_mean = _test_cv_classifier(clf, x, y, n=10)
        model = "Gaussian NB"
        
    elif classifier == DecisionTreeClassifier:
        clf = classifier(random_state=random_state)
        accuracy_score, roc_auc, pr_auc, precision, recall,\
        conf_matrix = _test_classifier(clf, x_train, y_train, x_test, y_test)
        scores, scores_mean = _test_cv_classifier(clf, x, y, n=10)
        model = "Decision Tree"
        
    
    elif classifier == CategoricalNB:
        clf = classifier(min_categories=min_categories)
        accuracy_score, roc_auc, pr_auc, precision, recall,\
        conf_matrix = _test_classifier(clf, x_train, y_train, x_test, y_test)
        scores, scores_mean = _test_cv_classifier(clf, x, y, n=10)
        model = "Categorical NB"

    else:
        print("Unknown classifier")
        
    results.append({
        'classifier:' : model,
        'Accuracy': round(accuracy_score, 3),
        'ROC AUC': round(roc_auc, 3),
        'PR AUC': round(pr_auc, 3),
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'Confusion matrix': _format_conf_matrix(conf_matrix),
        'CV scores': _format_cv_scores(scores),
        'CV mean score': round(scores_mean, 3)
    })
        
    results_df = pd.DataFrame(results)
    return results_df

