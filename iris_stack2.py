#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################## load packages #####################
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier


################## load data #####################
iris = datasets.load_iris()
x, y = iris.data[:, 1:3], iris.target


################## define classifier #####################
clf1 = KNeighborsClassifier(n_neighbors = 1)

clf2 = RandomForestClassifier(random_state = 1)

clf3 = GaussianNB()

lr = LogisticRegression()

sclf = StackingClassifier(classifiers = [clf1, clf2, clf3], 
                          use_probas=True,
                          average_probas=False,
                          meta_classifier = lr)


################## class result #####################
for clf, label in zip([clf1, clf2, clf3, sclf],
                      ['KNN',
                       'Random Forest',
                       'Naive Bayes',
                       'StackingClassifier']):
    
    scores = model_selection.cross_val_score(clf, x, y, cv = 3, scoring='accuracy')
    
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))