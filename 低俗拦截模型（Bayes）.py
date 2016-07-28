# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:30:57 2016
低俗拦截模型：贝叶斯
@author: Tomi
"""

import numpy as np
import pandas as pd
from time import time
from sklearn import metrics

def get_data():
    df = pd.read_excel(r"F:\textdata\sample.20160628(labelled).xlsx", encoding='gbk')
    label = df['label']
    content = df['content']
    return content, label

x, y = get_data()

corpus = np.array([line.strip() for line in x])
label = np.array(y)

from sklearn.feature_extraction.text import CountVectorizer
matrix = CountVectorizer().fit_transform(corpus)

from sklearn.cross_validation import train_test_split
print("Split data: train-test = 7:3")
x_train, x_test, y_train, y_test = train_test_split(matrix, label, test_size=0.3, random_state=42)

# Bayes模型
from sklearn.naive_bayes import MultinomialNB
print("Modeling NB...")
t0 = time()
clf = MultinomialNB().fit(x_train, y_train)
t1 = time()
print("Finish modeling(NB), costing %0.3fs" % (t1-t0))
# Bayes评估
p1 = clf.predict(x_test)
t2 = time()
single1 = clf.predict(x_test[0])
t3 = time()
print("Single Predict: %0.3fs" % (t3-t2))
print("NB Model Evaluate:")
accuracy = metrics.accuracy_score(y_test, p1)
print("accuracy_score: %f" % accuracy)
recall = metrics.recall_score(y_test, p1)
print("recall_score: %f" % recall)
print(metrics.classification_report(y_test, p1))
