"""
@author: Marcos
This creates bigData.csv and testData.csv
Unfortunately, since I am using a windows laptop,
paths are windows paths. Please edit them and run this file if you're testing the
Bernoulli NB implementation prior to running bernoulli.py
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

def uploadTrainData():
    path = 'C:\\Users\\User\\Documents\\COMP 551\\proj 2\\train\\pos\\'
    path1 = 'C:\\Users\\User\\Documents\\COMP 551\\proj 2\\train\\neg\\'
    files1 = [f for f in listdir(path) if isfile(join(path, f))]
    files2 = [f for f in listdir(path1) if isfile(join(path1, f))]
    
    col_name = ['class', 'text']
    nb_df = pd.DataFrame(columns = col_name)
    row_indx = 0
    save_path = "C:\\Users\\User\\Documents\\COMP 551\\proj 2\\Data.csv"
    for i in files1:
        fullFileName = path+i
        content = open(fullFileName,encoding="utf-8").read()
        nb_df.loc[row_indx]=[1,content]
        row_indx += 1

    for i in files2:
        fullFileName = path1+i
        content = open(fullFileName,encoding="utf-8").read()
        nb_df.loc[row_indx]=[0, content]
        row_indx += 1



#shuffle this so as to remove bias
    nb_df = nb_df.sample(frac=1)
#write to csv
    nb_df.to_csv('bigData.csv', sep='\t', encoding='utf-8', index = False)
    data = nb_df
    
    return data

def makeTest():
    path = 'C:\\Users\\User\\Documents\\COMP 551\\proj 2\\test\\'
    files1 = [f for f in listdir(path) if isfile(join(path, f))]
    
    col_name = ['text']
    nb_df = pd.DataFrame(columns = col_name)
    row_indx = 0
    for i in files1:
        fullFileName = path+i
        content = open(fullFileName,encoding="utf-8").read()
        nb_df.loc[row_indx]=[content]
        row_indx += 1
        
    nb_df.to_csv('testData.csv', sep='\t', encoding='utf-8', index = False)
    
    return