import nltk
import pymorphy2
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus.reader import WordListCorpusReader
import itertools
import pandas as pd
import string
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
###

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

class disamb():
    def __init__(self, fulldata, wordid):
        self.fulldata = fulldata
        self.wordid = wordid

        self.fulldata['sent'] = self.fulldata['sent'].apply( lambda s : s.lower())

        self.fulldata = self.fulldata[self.fulldata['word'] == self.wordid]

        self.X = fulldata["sent"].values

        self.Y = fulldata["word_id"]
        self.Y = self.Y.apply(lambda x: x.replace( self.wordid + "_", "", 1).upper())

        allpos = ["NOUN", "ADJF", "ADJS", "COMP", "VERB", "INFN", "PRTF", "PRTS", "GRND", "NUMR",
                  "ADVB", "NPRO", "PRED", "PREP", "CONJ", "PRCL", "INTJ"]

        self.Y = self.Y.apply(lambda x: allpos.index(x)).values

        #self.Yinit = self.Y
        #self.Yinit = self.Yinit.apply(lambda x: x.replace(self.wordid + "_", "", 1).upper())
        #self.Yinit = self.Y.apply(lambda x: allpos.index(x)).values

        print(self.fulldata)
        #print(self.Y)

    def createtokcorp(self):
        tokenizecorp = []
        #for c in corpus:
        #    tokenizecorp.append(nltk.word_tokenize(c))
        locs = []
        for index, row in self.fulldata.iterrows(): #loop over to find word location not character
            print(row)
            q = nltk.word_tokenize(row["sent"])
            tokenizecorp.append(q)
            start = row["start"]-1
            stop = row["stop"]-1
            word = row["sent"][start:stop] #extract the word
            locs.append(q.index(word))
        self.tokenizecorp = tokenizecorp
        self.locs = locs
