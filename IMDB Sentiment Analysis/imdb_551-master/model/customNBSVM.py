"""
@author: Marcos
Custom implementation of NBSVM as proposed by:
Wang, Sida. Manning, Christopher D. (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification.
"""

import numpy as np
import pandas as pd
import re  
import nltk  
nltk.download('stopwords')  
nltk.download('wordnet')
import pickle  
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC 
from sklearn.naive_bayes import MultinomialNB
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import cross_val_score, train_test_split

def preProcessing(data, test):
   
   #convert to numpy
    dataX = []
    Ytrain = []
    testData = []
    X = []
    testY = []
  
    for i, d in data.iterrows():
       dataX.append(d[1])
       Ytrain.append(d[0])

    for i, d in test.iterrows():
        testData.append(d[0])
    
    stemmer = WordNetLemmatizer()
    
    for sen in range(0, len(dataX)):  
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(dataX[sen]))
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # Converting to Lowercase
        document = document.lower()
        # Lemmatization
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        X.append(document)
        
    for sen in range(0, len(testData)):  
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(testData[sen]))
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # Converting to Lowercase
        document = document.lower()
        # Lemmatization
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        testY.append(document)
    
    return X, testY, Ytrain

def normalize(data):
    norm = np.sqrt((data ** 2).sum())
    return data / norm

def tfidf(documents):
    tfidfconverter = TfidfVectorizer(max_features=1500,ngram_range=(1,2), min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
    X = tfidfconverter.fit_transform(documents).toarray()  
    return X

def bigram_process(data):
	
	vectorizer = CountVectorizer(ngram_range=(1,2),max_features=15000, min_df = 5, stop_words=stopwords.words('english'))
	vectorizer = vectorizer.fit_transform(data).toarray()
	return vectorizer

def stochastic_descent(Xtrain, Ytrain, Xtest):
	
	clf = SGDClassifier(loss="hinge", penalty="l1", n_iter=20)
	print ("SGD Fitting")
	clf.fit(Xtrain, Ytrain)
	print ("SGD Predicting")
	Ytest = clf.predict(Xtest)
	return Ytest

def svm(Ynb, Ytest):
    
    clf = LinearSVC(class_weight=None, dual=False, fit_intercept=True,
     loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)  
    clf.fit(Ynb, Ytest)
    
    return clf

def mnb(Xtrain, Ytrain, Xtest):
    
    clf = MultinomialNB()  
    clf.fit(Xtrain, Ytrain)
    #Ytest = clf.predict(Xtest)
    return clf.fit(Xtrain, Ytrain)

def accuracy(Ytrain, Ytest):
	assert (len(Ytrain)==len(Ytest))
	num =  sum([1 for i, word in enumerate(Ytrain) if Ytest[i]==word])
	n = len(Ytrain)  
	return (num*100)/n





def main():
    #upload the data
    data = pd.read_csv('bigData.csv', sep = '\t',encoding = 'utf-8')
    test = pd.read_csv('testData.csv', sep = '\t',encoding = 'utf-8')
    
    #preprocess
    trainX, testY, Ytrain = preProcessing(data, test)
    
    #feauture extraction    
    XtrainBigram = bigram_process(trainX) #the feauteres using a bigram feauture extraction
    
    #split the data using the X train bigram data
    X_train, X_test, y_train, y_test = train_test_split(XtrainBigram, Ytrain, test_size=0.2, random_state=0) 
   
    #training diff models
    Y_bigram_output = stochastic_descent(X_train, y_train, X_test) #the predicted Y values using bi-gram with gradient descent
  
    #base learner is MNB
    #fit the nb model on the train sets
    
    nb_train_pred = mnb(X_train, y_train, X_test) #mnb predictions base
    #nb_test_pred = mnb(X_test, y_test, X_test)
    
    base_pred_train = nb_train_pred.predict(X_train) #the predicted output by NB
    base_pred_test = nb_train_pred.predict(X_test) #P base
    
   # (base_pred_train, y_train) #train svm on this
    #(base_pred_test, y_test) #predict on this
    
    #Meta learner is SVM
    y_trainArray = np.array(y_train)
    y_testArray = np.array(y_test)
    
    Y_nb_train = base_pred_train.reshape(-1,1)
    Y_nb_test = base_pred_test.reshape(-1,1)

    Y_svm_clf = svm(Y_nb_train, y_trainArray) #train the svm on (base_pred_train, y_train) 
    
    Y_svm_pred = Y_svm_clf.predict(Y_nb_test) #predict the model on base_pred_test
    
    
    #evaluate the model
    print ("Accuracy for the Bigram Model is ", accuracy(y_test, Y_bigram_output))
    print ("Accuracy for the Tri-gram Model with SVM is ", accuracy(y_test, Y_Trigram_output_svm ))
    print ("Accuracy for the Bi-gram Model with NBSVM is ", accuracy(y_test, Y_svm_pred)) #gives 87.14
    
    
    
if __name__ == '__main__':
    main()
