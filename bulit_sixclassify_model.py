from __future__ import print_function
import numpy as np
from time import time
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
#from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
#read feature data from user behavior file
def get_data(filename):
	data = load_svmlight_file(filename)
	return data[0], data[1]
#read data from file
print("loading data from file...")
t0 = time()
filename = "F:\\pythonFile\\textdata\\sample.20160628(labelled).txt"
X0,y0 = get_data(filename)
X = X0
y = np.array( [i for i in y0] )
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
t1 = time()
print("load data from file ok,time:%0.3fs" % (t1-t0))
X_single = X_test[0:1]
clf = RandomForestClassifier(n_estimators=100)
#pipeline = Pipeline([('clf', RandomForestClassifier(criterion='entropy'))])
#parameters = {'clf__n_estimators': (5, 10, 20, 50),'clf__max_depth': (30,50, 150),'clf__min_samples_split': (1, 2, 3,4),'clf__min_samples_leaf': (1, 2, 3,4)}
#clf = SVC() 
#grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='f1')
#grid_search.fit(X_train, y_train)
#print('best_f:%0.3f' % grid_search.best_score_)
#print('best:')
#best_parameters = grid_search.best_estimator_.get_params()
#for param_name in sorted(parameters.keys()):
#	print('\t%s: %r' % (param_name, best_parameters[param_name]))
#predictions = grid_search.predict(X_test)
#print(classification_report(y_test, predictions))

print("begin model fitting...")
t0 = time()
clf.fit(X_train,y_train)
t1 = time()
print("fit model ok,time:%0.3fs" % (t1-t0 ))
print( "testing model..." ) 
t0 = time()
pred = clf.predict(X_test)
t1 = time()
print("test model ok,time:%0.3fs" % (t1-t0))

accuracy_score  = metrics.accuracy_score(y_test,pred)
recall_score = metrics.recall_score(y_test,pred)
print("accuracy_score:%f" % accuracy_score)
print("recall_score:%f" % recall_score )
print( metrics.classification_report(y_test, pred) )
joblib.dump(clf, '/data/hadoop/apps/xieenming/antispam_model_filename_sixclassify/RandomForest_sixclass.pkl')
#predict single message 
t0 = time()
pred = clf.predict(X_single)
t1 = time()
print( pred )
print( "predict single message time:%fs" % (t1-t0) )
