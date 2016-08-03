# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from time import time
from sklearn import  metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def get_data(fileName):
    df = pd.read_excel(fileName)
    labels = df['label']
    content = df['content']
    return content,labels

x,y = get_data("F:\\textdata\\sample.20160628(labelled).xlsx")
# 去除数据首尾空格
text = np.array([line.strip() for line in x])
labels = np.array(y)
# 向量化数据
myMat = CountVectorizer().fit_transform(text)

print("Split data: train-test = 7:3")
x_train, x_test, y_train, y_test = train_test_split(myMat, labels, test_size = 0.3, random_state = 24)

# KNN模型
print("Modeling KNN...")
t0 = time()
clf = KNeighborsClassifier(n_neighbors= 20).fit(x_train, y_train)
t1 = time()
print("Finish modeling(KNN), cost %0.3fs" % (t1-t0))
# KNN评估
p1 = clf.predict(x_test)
t2 = time()
single1 = clf.predict(x_test[0])
t3 = time()
print("Single Predict: %0.3fs" % (t3-t2))
print("KNN model Evaluate:")
accuracy = metrics.accuracy_score(y_test, p1)
print("accuracy_score: %f" % accuracy)
recall = metrics.recall_score(y_test, p1)
print("recall_score: %f" %recall)
print(metrics.classification_report(y_test, p1))
print(metrics.confusion_matrix(y_test,p1))
