import nltk
import sys
import os
import pymorphy2
import re
import json
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from spacy.lang.ru import Russian
from nltk.corpus.reader import WordListCorpusReader
import itertools
import pandas as pd
import string
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.snowball import RussianStemmer
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from rnnmorph.predictor import RNNMorphPredictor
from pymystem3 import Mystem

class DisambiguationEngineException(Exception):
    def _init_ (self, *args):
        super()._init_(args)

class DisambiguationEngine():
    def __init__(self, fulldata, wordid, verbose = False):
        self.verbose = verbose
        self.fulldata = fulldata
        self.wordid = wordid
        #self.fulldata["sent"] = self.fulldata["sent"].apply( lambda x : x.replace( "[омонимия не снята]" , "") )
        #self.fulldata["sent"] = self.fulldata["sent"].apply( lambda x : x.replace( "[омонимия снята]" , "") )
        #self.fulldata["sent"] = self.fulldata["sent"].apply( lambda x : x.replace( r"\[ (.+?) * \(\d{4}\) .*?  \]" , "") )
        #self.fulldata["sent"] = self.fulldata["sent"].apply( lambda x: re.sub( r\[ (.+?) * \(\d{4}\) .*?  \]" , " ", x ) )
        #self.fulldata["sent"] = self.fulldata["sent"].apply( lambda x: re.sub( r"\[.*\d{+}\]", " ", x ) )
        self.fulldata["sent"] = self.fulldata["sent"].astype(str).apply(str.casefold)
        self.fulldata["word"] = self.fulldata["word"].astype(str).apply(str.casefold)
        self.fulldata = self.fulldata[self.fulldata["word"] == self.wordid]
        ##KKKKKKKKKKKKKKKKKKKKKIIIIIIIIIIIIIIIIIIIIIIIIWWWWWWWWWWWWWWWWWWWIIIIIIIIIIIIIIIIIIIIIIIII
        self.fulldata = self.fulldata.drop_duplicates(subset = ["sent"])
        self.X = self.fulldata["sent"].values
        target_word = self.fulldata["word_id"]
        target_word = target_word.apply(lambda x: x.replace( self.wordid + "_", "", 1).upper())
        self.allpos = ["NOUN", "ADJF", "ADJS", "COMP", "VERB", "INFN", "PRTF", "PRTS", "GRND", "NUMR",
                  "ADVB", "NPRO", "PRED", "PREP", "CONJ", "PRCL", "INTJ"]
        self.Y = target_word.apply(lambda x: self.allpos.index(x)).values #getting values corresponding to the correct part speech out of the labeled data
        self.target_word = target_word
        if self.verbose:
            print(self.X.shape)
            print(self.Y.shape)
    def CreateTokensCorpus(self):
        """
        Accepts: None
        Returns: 1 on success
        Description: Tokenizing sentences and extracting the target word from labeled sentences.
        Variables created: self.tokenizecorp and self.locs to use outside the class.
        Raises:
            - DisambiguationEngineException raised if fulldata is empty
        """
        tokenizecorp = []
        locs = []
        nlp = Russian()
        if self.fulldata.empty:
            raise DisambiguationEngineException("Lenght of fulldata is 0, cannot process.")
        for index, row in self.fulldata.iterrows(): #loop over to find target word location in each sentence
            try:
                doc = nlp(row["sent"])
                q = [token.text for token in doc ]
                #q = nltk.word_tokenize(row["sent"])
                start = row["start"]-1
                stop = row["stop"]-1
                word = row["sent"][start:stop] #extract the word
                q = [x for x in q if x not in (list(string.punctuation) ) ]
                locs.append(q.index(word))
                tokenizecorp.append(q)
            except Exception as e:
                print(row["sent"])
                print(f" the exception {str(e)} was raised while processing {row}")
                raise ValueError("row invalid")
        self.tokenizecorp = tokenizecorp
        print(len(tokenizecorp))
        self.locs = locs
        return 1
    def CreatePosCorpus(self):
        """
        Description: A Russian morphological analyzer is used to create the corpus of possible parts of speech for each word.
        This method loops over tokenized data, removing punctuation and special symbols that occur in Russian.
        This method then loops over the corpus of parts of speech to get possible parts of speech of a certain number of immediately preceeding and following words.
        Variables created: self.poscorp (all parts of speech), self.poscorpus (global context parts of speech) and self.poscorpus_mini (closest neighbors' parts of speech).
        """
        morph = pymorphy2.MorphAnalyzer()
        #morph.parse(repr(self.tokenizecorp))
        morph = RNNMorphPredictor(language="ru")
        #Tag = morph.TagClass
        self.poscorp = []
        for s in self.tokenizecorp:
            tokenizedsen = []
            for token in s:
                if token not in list(string.punctuation):
                    q = list(set( [p.pos for p in morph.predict(token) if p.pos != None ]  ))
                    tokenizedsen.append(q)
            self.poscorp.append( tokenizedsen )
        print(self.poscorp)
        look = 80
        self.poscorpus = []
        self.poscorpus.append( " ".join( self.allpos ) )
        for step in zip( self.locs, self.poscorp ):
            loc = step[0]
            tokenizedsen = step[1]
            #print(tokenizedsen)
            #print("KIWIKIWIKIWIKIWIKIWIKIWIKIWIKIWI")
            prevblock = tokenizedsen[ max(loc - look, 0 ) : loc ]
            nextblock = tokenizedsen[ (loc + 1) : min(loc + 1 + look, len(tokenizedsen))]
            #print(f"tokenizedsen {tokenizedsen}")
            #print(f"q {q}")
            print("BDUUUUUUUUUMMMMMMMMMMM")
            print(f"looking at preceding POS {prevblock}")
            print("BDUUUUUUUUUMMMMMMMMMMM")
            print(f"looking at succeeding POS {nextblock}")
            self.poscorpus.append(" ".join([ " ".join(q) for q in (nextblock + prevblock) ]))
            #self.poscorpus.append( " ".join(list(itertools.chain.from_iterable(prevblock))) + " " + " ".join(list(itertools.chain.from_iterable(nextblock))) )
        nlook = 5
        plook = 7
        self.poscorpus_mini = []
        self.poscorpus_mini.append( " ".join( self.allpos ) )
        for step in zip( self.locs, self.poscorp ):
            loc = step[0]
            tokenizedsen = step[1]
            prevblock = tokenizedsen[ max(loc - plook, 0 ) : loc ]
            nextblock = tokenizedsen[ (loc + 1) : min(loc + 1 + nlook, len(tokenizedsen)) ]
            self.poscorpus_mini.append( " ".join(list(itertools.chain.from_iterable(prevblock))) + " " + " ".join(list(itertools.chain.from_iterable(nextblock))) )
            #print("KIWIKIWIKIWIKIWIKIWIKIWIKIWIKIWI")
            #print(prevblock)
            #print("RAZDELRAZDELRAZDELRAZDELRAZDEL")
            #print(nextblock)
        if self.verbose:
            print(self.poscorpus_mini)

    def CreateNearestPosFeature(self):
        """
        Description: This method loops over the PoS corpus to find the distance (number of words in between) to each nearest PoS from the list of all parts of speech.
        It does this going both forward and backwards.
        """
        pos_distance_features_f = []
        pos_distance_features_b = []
        for step in zip( self.locs, self.poscorp ):
            loc = step[0]
            tokenizedsen = step[1]
            pos_distance_features_sent_f = []
            pos_distance_features_sent_b = []
            for pos_to_find in self.allpos:
                #print(pos_to_find)
                after_pos = tokenizedsen [loc + 1:]
                before_pos = tokenizedsen [:loc] [:: -1]
                first_pos_indexf = 1
                first_pos_indexb = 1
                for a in before_pos:
                    #print(a)
                    if pos_to_find in a:
                        #print("found it!")
                        break ##### Handle case when reaches the end of sent without finding pos (-1?)
                    first_pos_indexb += 1
                for a in after_pos:
                    #print(a)
                    if pos_to_find in a:
                        #print("found it!")
                        break ##### Handle case when reaches the end of sent without finding pos (-1?)
                    first_pos_indexf += 1
                pos_distance_features_sent_b.append(first_pos_indexb)
                pos_distance_features_sent_f.append(first_pos_indexf)
            #first_pos_indexb = 0
            #print(pos_distance_features_sent_b)
            pos_distance_features_b.append(pos_distance_features_sent_b)
            pos_distance_features_f.append(pos_distance_features_sent_f)
        #print(pos_distance_features_b)
        self.pos_distance_features_b = pos_distance_features_b
        self.pos_distance_features_f = pos_distance_features_f


    def CreateLemmaCorpus(self):
        """
        Description: A corpus of normal forms/ lemmas for each token is created.
        The corpus of lemmas is looped over to compute lemmas of a certain number of preceding and following words.
        Variables created: self.lemmacorp.
        Depends on: self.tokenizecorp, self.wordid.
        """
        morph = pymorphy2.MorphAnalyzer()
        lemmacorp = []
        for s in self.tokenizecorp:
            lemmasent = list(map(lambda q : morph.parse(q)[0].normal_form, s) )
            lemmacorp.append(lemmasent)
        self.lemmacorp = lemmacorp
        if self.verbose:
            print(lemmacorp)
        look = 15
        self.lemmaneighbors = []
        for step in zip( self.locs, self.lemmacorp):
            loc = step[0]
            lemmasent = step[1]
            prevblock = lemmasent[ max(loc - look, 0 ) : loc ]
            nextblock = lemmasent[ (loc +1) : min(loc + 1 + look, len(lemmasent))]
            self.lemmaneighbors.append(" ".join(prevblock + nextblock))
        if self.verbose:
            print(self.lemmaneighbors)

    def CreateRelativePositionEncoding(self):
        #print(self.X)
        #print(self.tokenizecorp)
        morph = pymorphy2.MorphAnalyzer()
        self.rpec = []
        #self.X_labeled
        #vectorizer = TfidfVectorizer()
        #Xtrans = vectorizer.fit_transform(self.X)
        look = 4
        for step in zip( self.locs, self.tokenizecorp ):
            loc = step[0]
            tokenizedsen = step[1]
            #print(tokenizedsen)
            #print("KIWIKIWIKIWIKIWIKIWIKIWIKIWIKIWI")
            m = Mystem()
            #lemmas = m.lemmatize(text)
            prevblock = tokenizedsen[ max(loc - look, 0 ) : loc ]
            nextblock = tokenizedsen[ (loc + 1) : min(loc + 1 + look, len(tokenizedsen))]
            prevblock = list(map( lambda x: f"{prevblock[::-1].index(x)}_{x}", prevblock ))
            nextblock = list(map( lambda x: f"{nextblock[::-1].index(x)}_{x}", nextblock ))
            #nextblocklemma = list(map(lambda q : morph.parse(q)[0].normal_form, nextblock) )
            #prevblocklemma = list(map(lambda q : morph.parse(q)[0].normal_form, prevblock) )
            nextblocklemma = list(map(lambda q : m.lemmatize(q)[0], nextblock) )
            prevblocklemma = list(map(lambda q : m.lemmatize(q)[0], prevblock) )
            self.rpec.append("".join([ "".join(q) for q in (nextblocklemma + prevblocklemma) ]))
        print(self.rpec)
    def CreateFeatures(self):
        """
        Description: creating features and fitting with 3 types of sentence vectorizers.
        Features:
            - Global context words: fitting with TfidfVectorizer - Xtrans
            - Global context parts of speech: fitting with HashingVectorizer - Xpos2
            - Local context parts of speech: fitting with CountVectorizer - Xpos1
            - Local context lemmas: fitting with CountVectorizer - Xlemma
        Variables created: self.Xtot (concatenation of all features)
        """
        vectorizer = TfidfVectorizer()
        Xtrans = vectorizer.fit_transform(self.X)
        ##############################
        pos2vectorizer = HashingVectorizer(n_features=2**8, ngram_range=(1,2))
        Xpos2 = pos2vectorizer.fit_transform(self.poscorpus)
        Xpos2 = Xpos2.toarray()
        Xpos2 = Xpos2[1:]
        #############################
        pos1vectorizer = CountVectorizer()
        Xpos1 = pos1vectorizer.fit_transform(self.poscorpus_mini)
        Xpos1 = Xpos1.toarray()
        Xpos1 = Xpos1[1:]
        ##############################
        vect = CountVectorizer()
        Xlemma = vect.fit_transform(self.lemmaneighbors)
        Xlemma = Xlemma.toarray()
        ###############################
        rpecvectorizer = CountVectorizer()
        #rpecvectorizer = HashingVectorizer(n_features=2**8, ngram_range=(1,2))
        rpec = rpecvectorizer.fit_transform(self.rpec)
        rpec = rpec.toarray()
        ###############################
        self.Xtot = np.concatenate((Xlemma, Xpos2, rpec), axis = 1)
        #self.Xtot = np.concatenate((Xlemma, Xpos1, Xpos2, rpec), axis = 1)
        #self.Xtot = np.concatenate((Xtrans.toarray(), rpec), axis = 1)
        self.corr = pd.DataFrame(self.Xtot).corr()
        #print(self.corr)
        self.corr.to_csv("/Users/biatris/Desktop/Project/results_correlation.tsv")
        #self.Xtot = np.concatenate((self.pos_distance_features_b, self.pos_distance_features_f), axis = 1)
        #self.Xtot = np.concatenate((Xpos2), axis = 1)
        #self.Xtot = np.concatenate((Xpos2, Xlemma), axis = 1) # only using two due to the current feauture-data ratio
        #self.Xtot = rpec
        if self.verbose:
            print(Xpos1.shape, Xpos2.shape, Xlemma.shape)
        print("KKKKKKKKKIIIIIIIIIIWWWWWWWWWWIIIIIIIII")
        print(self.Xtot.shape)
    def CreateModel(self):
        """
        Creating a model using SGD classifier.
        """
        if self.verbose:
            print(self.mysvc.coef_.tolist())
        mysgd = SGDClassifier(loss="hinge", penalty="elasticnet", tol=1e-5, n_jobs=-1, max_iter=10000, eta0=0.0000001, alpha=1e-3)
        print(self.Xtot.shape, self.Y.shape)
        mysgd.fit(self.Xtot, self.Y)

    def SgdScore(self, rounds):
        """
        Returns: the mean of the array elements and the spread of a distribution.
        Description: performing cross-validation on score. Randomly splitting fulldata into train and test data sets of indicated proportions.
        Setting a number of rounds for cross-validation training. Fitting data. Computing results mean and distribution.
        """
        res = []
        for k in range(rounds):
            X_train, X_test, y_train, y_test = train_test_split(self.Xtot, self.Y, test_size=0.2)
            mysgd = SGDClassifier(loss="hinge", penalty="elasticnet", tol=1e-5, n_jobs=-1, max_iter=10000, eta0=0.0000001, alpha=1e-3)
            mysgd.fit(X_train, y_train)
            res.append(mysgd.score( X_test, y_test ) )
        return (round(np.mean(res),5), " +/-", round(np.var(res), 5 ))

    def FindErrors(self):
        """
        Returns: Confusion matrix for the POS prediction
        Description: Writes results into dataframe and generates confusion matrix.
        """
        X_aug = self.Xtot
        X_aug = np.c_[self.Xtot, self.fulldata[["sent"]].values]
        #print(X_aug)
        X_train, X_test, y_train, y_test = train_test_split(X_aug, self.Y, test_size=0.2)
        X_test_sent = X_test[:,-1:][:,0]
        X_test = X_test[:,:-1]
        X_train = X_train[:,:-1]
        mysgd = SGDClassifier(loss="hinge", penalty="elasticnet", tol=1e-5, n_jobs=-1, max_iter=10000, eta0=0.0000001, alpha=1e-3)
        mysgd.fit(X_train, y_train)
        res = pd.DataFrame()
        res["sent"] = X_test_sent
        res["ground_truth"] = y_test
        pred = mysgd.predict(X_test)
        res["preds"] = pred
        res["success"] = (res["preds"] == res["ground_truth"])
        Q = confusion_matrix(y_test, pred)
        return (Q, res)
