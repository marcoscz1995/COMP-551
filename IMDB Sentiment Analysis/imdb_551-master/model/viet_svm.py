"""
@author: Viet
uni/bi-gram linearSVM using 10fold cross validation
"""

from model.pre2 import retrieve_and_pre, dump_to_csv
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score


def main():
    X, y, test, names = retrieve_and_pre(fromsave=True, tfidfpca=True, n_components=100000)
    y = np.asarray(y)
    #print("test shape", test.shape)
    #print(test[0])
    print(X.shape[0], y.shape[0])
    assert X.shape[0] == y.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

    clf = LinearSVC(C=0.5, verbose=1, max_iter=2000)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    scores = cross_val_score(clf, X_train, y_train, cv=10)
    for i in range(len(scores)):
        print("Score on %i th fold: %f" % (i, scores[i]))
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    #print("Accuracy score: %f%%" % (scr * 100))

    f1_scr = f1_score(y_test, pred)
    print("F1 score: %f%%" % (f1_scr*100))
    dump_to_csv(clf, test, names, 'autosvm')

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print("true neg: %i\tfalse pos: %i\nfalse neg: %i\ttrue pos: %i\n" % (tn, fp, fn, tp))

    print("precision: %f; recall: %f" % ((tp/(tp+fp)), (tp/(tp+fn))))

if __name__ == '__main__':
    main()
