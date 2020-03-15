# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:17:56 2020

@author: gmnya
"""

# Import libraries
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from util import plot_roc
import pickle


class NLPModel(object):
    
    def __init__(self):
        self.clf = MultinomialNB()
        self.vectorizer = TfidfVectorizer()
        
    def vectorizer_fit(self, X):
        self.vectorizer.fit(X)
        
    def vectorizer_transform(self, X):
        X_transformed = self.vectorizer.transform(X)
        return X_transformed
    
    def train(self, X, y):
        self.clf.fit(X,y)
        
    def predict_proba(self, X):
        y_proba = self.clf.predict_proba(X)
        return y_proba[:, 1]
    
    def predict(self, X):
        y_pred = self.clf.predict(X)
        return y_pred
    
    def pickle_vectorizer(self, path='models/TFIDFVectorizer.pkl'):
        """
        Save the trained vectorizer for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            print("Pickled vectorizer at {}".format(path))
            
    def pickle_clf(self, path='models/SentimentClassifier.pkl'):
        """Saves the trained classifier for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Pickled classifier at {}".format(path))
            
    def plot_roc(self, X, y):
        """ Plot the ROC curve for X_test and y_test."""
        plot_roc(self.clf, X, y)