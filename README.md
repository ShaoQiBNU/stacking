集成学习Stacking方法详解
======================

# 一. stacking方法介绍

> stacking是用初始训练数据学习出若干个基学习器后，将这几个学习器的预测结果作为新的训练集，来学习一个新的学习器。具体过程如下：

## 1. 数据划分和基学习器

> 将数据集划分为训练集和测试集，这里采用三个基学习器，分别为XGBoost、RandomForest和KNN，基学习器在学习过程中，会采用k折交叉验证，这里以5折交叉验证为例（将训练集划分为5份）。第二步的融合模型常采用Logistic Regression模型。如图所示

img1111

## 2. stacking过程

> stacking过程分为两步，第一步是基学习器训练数据集得到第二步的训练数据，第二步基于第一步的训练数据采用Logistic Regression模型学习训练，得到最终结果。具体如下：

### (1) 基学习器得到训练集

> 以XGBoost为例说明，此处采用的是5折交叉验证，所以先将训练数据集Training Data划分为5份training1、training2、training3、training4和training5。之后将其中4折作为训练集，1折作为测试集，XGBoost模型基于4折数据训练，对1折数据进行预测，同时对整个测试集Testing Data做预测，这样的过程重复5次，会得到5份training的predict数据和Testing Data的predict数据，然后将5份training的predict数据纵向叠起来得到基学习器的Training Data learner，5份Testing Data的predict数据取平均值得到基学习器的Testing Data learner，如图所示：

img22222

> 随机森林和KNN同XGBoost，也进行上述过程。最后，将三个基学习器的Training Data learner横向拼接在一起，得到第二步的训练集Training Data learners，将三个基学习器的Testing Data learner横向拼接在一起，得到第二步的测试集Testing learners，如图所示：

ing3333

### (2) 融合训练和预测

> 基于(1)得到的训练集和测试集，采用Logistic Regression模型对训练集Training Data learners进行训练，然后对测试集Testing learners进行预测，得到预测结果。当对新的测试集进行预测时，测试集需要先经郭所有基学习器预测，然后横向拼接得到测试集，最后再对测试集进行预测得到测试结果，如图所示：

ing4444

# 二. Stacking实例

> stacking的使用方法主要有以下三种，选取KNN，Random Forest和GaussianNB模型作为基学习器，lr模型作为融合模型，3折交叉验证对iris数据进行分类，具体如下：

# 1. 基学习器的特征输出作为融合模型的输入

> stacking的基本使用方法，使用基学习器的特征输出作为融合模型的输入，代码如下：

```python
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
```

# 2. 基学习器的类别概率值作为融合模型的输入

> 使用基学习器产生的类别概率作为融合模型的输入，这种情况下需要将StackingClassifier的参数设置为 use_probas=True。另外，需要注意一下 average_probas的设置，如果设置average_probas=True，那么这些基学习器对每一个类别产生的概率值会被平均，如果设置average_probas=False，那么这些基学习器对每一个类别产生的概率值会被拼接。

```
假设有两个基学习器产生的概率输出为：

classifier 1: [0.2, 0.5, 0.3]
classifier 2: [0.3, 0.3, 0.4]

average_probas=True
融合模型的训练数据: [0.25, 0.45, 0.35]

average_probas=False
融合模型的训练数据: [[0.2, 0.3, 0.3], [0.3, 0.4, 0.4]]
```

> 代码如下：

```python
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
```

# 3. 分特征

> 对训练数据中的特征维度进行操作，这次不是给每一个基学习器全部的特征，而是给不同的基学习器分不同的特征，即比如基学习器1训练前半部分特征，基学习器2训练后半部分特征（可以通过sklearn 的pipelines 实现），最终通过StackingClassifier组合起来。

```python
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
```


