#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################## load packages #####################
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline


################## load data #####################
iris = datasets.load_iris()
x, y = iris.data, iris.target


################## define classifier #####################

pipe1 = make_pipeline(ColumnSelector(cols=(0, 1)),
                      LogisticRegression())
pipe2 = make_pipeline(ColumnSelector(cols=(2, 3)),
                      LogisticRegression())

sclf = StackingClassifier(classifiers=[pipe1, pipe2], 
                          meta_classifier=LogisticRegression())

################## fit and predict #####################
sclf.fit(x, y)

print(sclf.predict(x))

########### predict class probability ###########
print(sclf.predict_proba(x))