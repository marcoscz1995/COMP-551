"""
@author: Marcos
#accuracy = 81.28%
Scratch implementation of Bernoulli NB
NOTE: Use trainTestMaker.py to create bigData.csv and testData.csv first
"""

import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy import genfromtxt
import csv
import nltk
import re
nltk.download("stopwords")
from nltk.corpus import stopwords
from functools import reduce
import operator

def token():
    
    data = pd.read_csv('bigData.csv', sep = '\t',encoding = 'utf-8')
    test = pd.read_csv('testData.csv', sep = '\t',encoding = 'utf-8')
    
    #some cleaning
    data['text'] = data['text'].str.lower().str.split(' ') 
    stop = stopwords.words('english')
    data['text'] = data['text'].apply(lambda x: [item for item in x if item not in stop])
    
    test['text'] = test['text'].str.lower().str.split(' ') 
    test['text'] = test['text'].apply(lambda x: [item for item in x if item not in stop])
    
    return data, test

def vectorize(data, test):
    Xpos = []
    Xneg = []
    Xtest = []
    for i,d in data.iterrows():
        if d['class'] ==1:
            Xpos.append(d['text'])
        else:
            Xneg.append(d['text'])
            
    for i,d in test.iterrows():
        Xtest.append(d['text'])
            
    return Xpos, Xneg, Xtest


def vocabulary(data, switch):      
    #word count
    #find the frequency of each word per document
    V= {}
    for i, row in data.iterrows():
        for word in set(row["text"]):
            if word in V:#we only want to count a word once per document
                V[word] += 1
            else: 
                V[word] = 1    
    if switch == 1:
        V = sorted(V, key = V.__getitem__, reverse = True)[:1000]
        return V
                
    V = np.array(list(V.keys()), dtype = str)
    
    return V


def matrix(Xneg,Xpos,test, V):    #Xneg is every document and its unique words 
    BNeg = []    #make this faster
    BPos = []
    Btest = []
    for d in Xneg:
        BNeg.append((np.isin(V, d)).astype(int))
    for d in Xpos:
        BPos.append((np.isin(V, d)).astype(int))
    for d in test:
        Btest.append((np.isin(V, d)).astype(int))
        
    return BPos, BNeg, Btest


def thetas(data):
    #prior probabilites
    y1=0
    for d in data["class"]:
        if d == 1:
            y1+=1
    #number of examples where y = 0
    y0=0
    for d in data["class"]:
        if d == 0:
            y0+=1
            
    theta0 = (y0/len(data)) #prior probability  P(y)
    theta1 = (y1/len(data))
    
    return theta0, theta1

def wordCount(V, Xneg, Xpos):      
    #number of documents of each class that contains word i
    Ponly= {}
    Nonly= {}
    
    #initialize the dictionary
    for word in V:
        if word not in Ponly:
            Ponly[word] = 1
            Nonly[word] = 1
    
    #fill the dictionary
    for word in V:
        for doc in Xneg:
            if word in doc:
                Nonly[word] += 1
        for doc in Xpos:
            if word in doc:
                Ponly[word]+=1
                
    return Ponly, Nonly


def lengths(data, Xpos, Xneg):
    return(len(data), len(Xpos), len(Xneg))

def conditionalProbabilities(N1, N0, Pcount, Ncount):    
    return({k: (Pcount[k]+1)/(N1+2) for k in Pcount},{k: (Ncount[k]+1)/(N0+2) for k in Ncount})     #p0, p1

def classifier(Btest, px1, px0, theta0, theta1, length): ####finish this
    
    docID= (list(range(1, length+1))) #add 1 to length
    
    #get the probabilities from the dictionaries as arrays
    px0 = np.matrix(list(px0.values()))
    px1 = np.matrix(list(px1.values()))
    #convert to matrices
    Btest = np.matrix(Btest)   
    #find product
    prob0 = theta0*(np.product((np.multiply(Btest, px0) + np.multiply((1 - Btest),(1-px0))), axis = 1)) 
    prob1 = theta1*(np.product((np.multiply(Btest, px1) + np.multiply((1 - Btest),(1-px1))), axis = 1))
    
    y = np.greater_equal(prob1,prob0).astype(int)
    y = reduce(operator.concat, y.tolist())
    
    return np.column_stack((docID, y))

def accuracy(Ytrain, Ytest):
	assert (len(Ytrain)==len(Ytest))
	num =  sum([1 for i, word in enumerate(Ytrain) if Ytest[i]==word])
	n = len(Ytrain)  
	return (num*100)/n

def validation(V, data_train, data_valid, y_valid):
    Xpos, Xneg, Xvalid = vectorize(data_train, data_valid ) #documents from pos and neg reviews. this is for the validation
    Bpos, Bneg, Btest = matrix(Xneg, Xpos, Xvalid, V)  #get the training and test matrices
    theta0, theta1 = thetas(data_train) #prior probabilities
    N, N1, N0 = lengths(data_train, Xpos, Xneg) #lengths of pos and neg docs
    Pcount, Ncount = wordCount(V, Xneg, Xpos) #word frequencies
    px1, px0 = conditionalProbabilities(N1,N0,Pcount,Ncount) #conditional probabilities
    y_valid_pred = classifier(Btest,px1,px0, theta0, theta1, len(Xvalid)) #final solution. Doc ID and its class
    
    print ("Accuracy for the Bigram Model is ", accuracy(y_valid, y_valid_pred[:,1]))
    return y_valid_pred[:,1]

def precision(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    len1 = len(y_pred)
    if len1 == 0:
        return 0
    else:
        return len(i) / len1
    
def recall(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    return len(i) / len(y_true)

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0
    else:
        return 2 * (p * r) / (p + r)

def testSet(V, data_train, data_valid):
    Xpos, Xneg, Xvalid = vectorize(data_train, data_valid) #documents from pos and neg reviews. this is for the validation
    Bpos, Bneg, Btest = matrix(Xneg, Xpos, Xvalid, V)  #get the training and test matrices
    theta0, theta1 = thetas(data_train) #prior probabilities
    N, N1, N0 = lengths(data_train, Xpos, Xneg) #lengths of pos and neg docs
    Pcount, Ncount = wordCount(V, Xneg, Xpos) #word frequencies
    px1, px0 = conditionalProbabilities(N1,N0,Pcount,Ncount) #conditional probabilities
    
    return classifier(Btest,px1,px0, theta0, theta1, len(Xvalid))

def main():
    #uploads data and test and tokenizes it
    #note that the data is in CSV form, is already shuffled
    data1, test1 = token() 
    
    #split into train and validation set
    data_train = data1[:20000] #train set
    data_valid = data1[20000:] #validation set
    y_valid=data_valid['class']
    
    #make vocabulary
    V = vocabulary(data_train, 1)  #if 1, use only top 1000 words, if 0 use all words
    
    #print accuracy and get models prediction on the validation set
    y_valid_pred = validation(V, data_train, data_valid, y_valid)
    
    #get the models prediction on the test set
    y_test_pred = testSet(V, data1, test1)
    
    #metrics
    print(precision(np.array(y_valid), np.array(y_valid_pred)))
    print(recall(y_valid, y_valid_pred))
    
    print(f1(y_valid, y_valid_pred))
    
if __name__ == '__main__':
    main()


