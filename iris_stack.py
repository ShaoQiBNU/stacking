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

sclf = StackingClassifier(classifiers = [clf1, clf2, clf3], meta_classifier = lr)


################## class result #####################
for clf, label in zip([clf1, clf2, clf3, sclf],
                      ['KNN',
                       'Random Forest',
                       'Naive Bayes',
                       'StackingClassifier']):
    
    scores = model_selection.cross_val_score(clf, x, y, cv = 3, scoring='accuracy')
    
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))
    
    
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools

gs = gridspec.GridSpec(2, 2)

fig = plt.figure(figsize=(10,8))

for clf, lab, grd in zip([clf1, clf2, clf3, sclf], 
                         ['KNN', 
                          'Random Forest', 
                          'Naive Bayes',
                          'StackingClassifier'],
                          itertools.product([0, 1], repeat=2)):

    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf)
    plt.title(lab)
