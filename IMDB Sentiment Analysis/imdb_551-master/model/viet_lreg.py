"""
@author: Viet
A pretty standard logistic regression model with saga solver, l2 regularization, and 10fold xval.
"""
from model.pre2 import retrieve_and_pre, dump_to_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def main():
    X, y, test, names = retrieve_and_pre(fromsave=True, n_components=50000)
    y = np.asarray(y)
    assert X.shape[0] == y.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)
    clf = LogisticRegression(penalty='l2', solver='saga', verbose=True)

    scores = cross_val_score(clf, X_train, y_train, cv=10)
    for i in range(len(scores)):
        print("Score on %i th fold: %f" % (i, scores[i]))
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    # print("Accuracy score: %f%%" % (scr * 100))

    clf.fit(X_train, y_train)
    pred = clf.fit(X_test)
    f1_scr = f1_score(y_test, pred)
    print("F1 score: %f%%" % (f1_scr * 100))
    dump_to_csv(clf, test, names, 'autosvm')

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print("true neg: %i\tfalse pos: %i\nfalse neg: %i\ttrue pos: %i\n" % (tn, fp, fn, tp))

if __name__ == '__main__':
    main()
