import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.linear_model import LinearRegression
import math
from sklearn.metrics import mean_squared_error, r2_score
from nltk import ngrams
import re
import time

def printdict(d):
    'A function to print a dictionary in a readable fashion'
    if(isinstance(d, dict)):
        for key,value in d.items():
            print(str(key) + " : " + str(value))
    elif(isinstance(d, list)):
        for element in d:
            printdict(element)
    else:
        print("not a dict.")

def preprocessing(data,ngr=False):
    'The preprocessing function of our program'
    X = []
    y = []
    is_root = []
    controversiality = []
    children = []
    wordfreq = {}

    # first pass: build features
    for element in data:
        if element["is_root"] == False:
            is_root.append(0)
        else:
            is_root.append(1)

        #build controversiality
        controversiality.append(element["controversiality"])
        children.append(element['children'])
        # processing y
        y.append(element["popularity_score"])

    # preprocessing text
    # We will attempt to filter out stopwords and see if it improves our model
    #stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
        stop_words = list()
        if ngr:
            element["text"] = re.sub(r'[^\w\s]', '', element["text"]).encode('ascii', 'ignore').lower().decode().split(" ")
            element["text"] = " ".join(element["text"])
            element["text"] = " ".join(element["text"]).split()
            bigrams = ngrams(element["text"], 2)
            for e in bigrams:
                if e in stop_words:
                    continue
                if e not in wordfreq:
                    wordfreq[e] = 1
                else:
                    wordfreq[e] += 1
        else:
            element["text"] = element["text"].encode('ascii', 'ignore').lower().decode().split(" ")
            try:
                for e in element["text"]:
                    if e in stop_words:
                        continue
                    if e not in wordfreq:
                        wordfreq[e] = 1
                    else:
                        wordfreq[e] += 1
            except:
                print("set is not in the desired format.")
                return
    # we build wordfreq, consisting of 160 different words
    wordfreq = sorted(wordfreq.items(), key=lambda a: a[1], reverse=True)[:161]
    del(wordfreq[6]) #remove the empty character
    print(wordfreq[:7])
    # convert word freq tuple to just words, and append to feature matrix
    wordfreq = [x[0] for x in wordfreq]
    wordlen = list()
    for element in data:
        words = element["text"]
        #element["text"] = " ".join(element["text"])
        #element["text"] = " ".join(element["text"]).split()
        #words = ngrams(element["text"], 2)
        wordlen.append(len(words))
        bow_vect = np.zeros(len(wordfreq))
        for word in words:
            for i, bow_word in enumerate(wordfreq, start=0):
                if word == bow_word:
                    bow_vect[i] += 1
        X.append(bow_vect)
    X = np.asarray(X)

    X = np.append(X, np.asarray(is_root)[:, None], axis=1)
    X = np.append(X, np.asarray(controversiality)[:, None], axis=1)
    X = np.append(X, np.asarray(children)[:, None], axis=1)
    X = np.append(X, np.asarray([float(i) / max(wordlen) for i in wordlen])[:, None], axis=1)

    controroot = []
    rootchild = []
    comlenisrootchildren = []
    for element in data:
        controroot.append(element["controversiality"]*element["is_root"])
        rootchild.append(element["is_root"]*element["children"])
        comlenisrootchildren.append(element['is_root']*element['children']*len(element['text']))
    X = np.append(X, np.asarray(controroot)[:, None], axis=1)
    X = np.append(X, np.asarray(rootchild)[:, None], axis=1)
    X = np.append(X, np.asarray(comlenisrootchildren)[:, None], axis=1)

    f = open('words.txt', 'w+')
    for w in wordfreq:
        f.write(w)
        f.write('\n')
    f.close()
    # add bias
    X = np.concatenate((np.ones(shape=(np.shape(X)[0],1)), X), axis=1)
    y = np.asarray(y)
    print(X.shape)
    return X, y

def train_test_split(X, y):
    'Splits matrices X, y into train,val, test sets according to guidance'
    return X[:10000], X[10000:11000], X[11000:], y[:10000], y[10000:11000], y[11000:]

class LinReg:
    'This is the linear regression class used in this assignment'
    def __init__(self, X):
        'Initializes the essentials'
        self.weights = np.random.normal(0, 1e-4, size=np.shape(X)[1])
        self.mselist = []
        self.cost = []

    def lse(self, X, y):
        'closed form solution: performs matrix stuff on training set'
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    def bgd(self, X, y, decay=0.99, initlr=1e-5, threshold=1e-3):
        'batch gradient descent: does exactly what it sounds like it does'
        lr = initlr/(1+decay)
        self.weights2 = self.weights - 2*lr*(np.dot(X.T, self.predict(X)-y))
        i = 1
        #while np.linalg.norm(self.weights2 - self.weights) > threshold:

        while np.abs(self.J(np.dot(X, self.weights2), y) - self.J(np.dot(X, self.weights),y)) > threshold:
            self.mselist.append(self.mse(np.dot(X, self.weights), y))
            self.cost.append(self.J(np.dot(X, self.weights), y))
            #print(str.format('{0:.6f}', np.linalg.norm(self.weights2 - self.weights)))
            print("∆cost: " + str(np.abs(self.J(np.dot(X, self.weights), y) - self.J(np.dot(X, self.weights2),y))))
            self.weights = self.weights2
            lr = initlr / (1 + decay*i)
            i += 1
            self.weights2 = self.weights - 2 * lr * (np.dot(X.T, self.predict(X) - y))

    def J(self, hyp, y):
        return 1/2/np.shape(y)[0]*sum(np.square(y-hyp))

    def predict(self, X):
        'Give a matrix X, returns hypothesis(x) of our model'
        return np.dot(X, self.weights)

    # metrics

    def r2_scr(self, hyp, y):
        'Calculates the '
        y_avg = np.full(shape=np.shape(y),fill_value=1/np.shape(y)[0]*sum(y))
        e = y - hyp
        SST = sum(np.square(y - y_avg))
        SSR = sum(np.square(e))
        r2 = 1 - SSR/SST
        return 1-(1-r2)*(hyp.shape[0]-1)/(hyp.shape[0]-67-1)

    def mse(self, hyp, y):
        return 1/np.shape(y)[0]*sum(np.square(y - hyp))

def main():
    #open json
    with open("proj1_data.json") as fp:
        rawdata = json.load(fp)
    """
    data = sorted(rawdata, key=lambda d: d["popularity_score"],reverse=True)
    for element in range(10):
        print(data[element])
    """
    X, y = preprocessing(rawdata)
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y)


    reg = LinReg(X_train)
    start = time.time()
    reg.lse(X_train, y_train)
    #reg.bgd(X_train, y_train, decay=2, initlr=1e-8, threshold=1e-1)
    end = time.time()
    print("time that took: %s" % (end - start))
    pred = reg.predict(X_val)
    test_pred = reg.predict(X_test)
    pred_train = reg.predict(X_train)
    train_mse = reg.mse(pred_train, y_train)
    print()
    print()
    print("Training MSE: %s " % train_mse)
    print("Validation MSE: %s " % reg.mse(pred, y_val))
    print("R^2: %s " % reg.r2_scr(pred, y_val))
    print("Test MSE: %s " %reg.mse(test_pred, y_test))
    print("Test R^2: %s " % reg.r2_scr(test_pred, y_test))
    """"
    plt.subplot(reg.mselist)
    plt.title("MSE Over time")
    plt.xlabel("iterations")
    plt.ylabel("MSE")
    """
    plt.plot(reg.cost)
    plt.xlabel("iteration")
    plt.ylabel("quadratic loss")
    plt.show()

    """
    plt.scatter(y_val, pred)
    plt.xlabel("Popularity score: $Y_i$")
    plt.ylabel("Predicted popularity score: $\hat{Y}_i$")
    plt.title("score vs predicted score: $Y_i$ vs $\hat{Y}_i$")
    x = np.linspace(0, 10, 1000)
    plt.plot(x,x, "--r")
    plt.show()
    """


if __name__ == '__main__':
    main()
