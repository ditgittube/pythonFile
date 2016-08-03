# coding: utf-8
#
# model: train multiclass(0-7) classification model with bag of words
# description: 
#       this program train multiclass(0-7) classification model based on bag of words


from __future__ import print_function
from pprint import pprint
from optparse import OptionParser

import numpy as np
import sys
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile,chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.externals import joblib

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression


reload(sys)
sys.setdefaultencoding('utf-8')

# parse commandline arguments
op = OptionParser()
op.add_option("-d","--deploy",
              action="store_true", dest="deploy",
              help="generate model file to deploy")
op.add_option("-t","--train",
              action="store_true", dest="train",
              help="train model and print performance for test dataset")
op.add_option("-f","--rf",
              action="store_true", dest="rf",
              help="generate random forest model file to deploy")

(opts,args) = op.parse_args()



#read binary corpus and labels
def read_sample(file='dirty-train'):
	print("begin to read binary corpus and labels...\n")
	t0 = time()
	data_path = 'data/dirty_sample/' 
	corpus = np.array( [ line.strip() for line in open(data_path+file+'-words',"r")] )
	n=1
	labels=[]
	for line in open(data_path+file+'-labels','r'):
		try:
			label=int(line.strip() )
			if label not in [0,1,2,3,4,5,6,7]:
				print(label)
				print(n)
			labels.append(int(line.strip()) )
			n=n+1
		except 	Exception, e:
			print(e)
			print(n)
			sys.exit(1)
	labels=np.array(labels)

	print( "read corpus and labels ok,time:%f\n\n" % (time()-t0) )
	return corpus,labels



#function name:
#          get_vocabulary()
#description:
#          fit the corpus and establish the dict


def get_vocabulary(select=10,output=False):
		print("begin to fit the corpus and establish the vocabulary\n")
		t0=time()

		vect=CountVectorizer(min_df=1,max_df=0.5,ngram_range=(1,1),binary=False).fit(corpus)

		#get a vocabulary with words in samples(it is a iterable list)
		dic_self=vect.get_feature_names()

		#transform the samples to the term-documnet matrix
		tdm=vect.fit_transform(corpus)
		print("there are %d samples\n " %len(corpus))
		print("the size of term -document matrix is ",tdm.shape,"\n")
		print("the vocabulary has %d words\n" %len(dic_self))

		#select  the features by chi2
		sel_f=SelectPercentile(chi2,select)
	#	sel_f=SelectKBest(chi2,100000)

		tdm_out=sel_f.fit_transform(tdm,labels)
		print("the size of selected term-document matrix is ",tdm_out.shape,"\n")
		index=sel_f.fit(tdm,labels).get_support(indices=False)

		#create a new vocabulary having  selected words of the old one
		dic_select=[]
		dic_unselect=[]
		n=0
		for i in index:
			if i:
				dic_select.append(dic_self[n])
			else:
				dic_unselect.append(dic_self[n])
			n=n+1


		print("vocabulary has been established with  %d words, cost time: %f\n" %(len(dic_select),(time()-t0)) )
		
		if output:
			out_file=open("dic_select",'w')
			for i in dic_select:
				out_file.write(i)
				out_file.write("\n")
			out_file.close()

			out_file=open("dic_unselect",'w')
			for i in dic_unselect:
				out_file.write(i)
				out_file.write("\n")
			out_file.close() 
		
		return dic_select

#function name:  
#			get_vocabulary_L2()
#description:
#			get the dictionary by L2
def get_vocabulary_L2():
	vect=CountVectorizer(min_df=1,max_df=0.5)
	tdm=vect.fit_transform(corpus)

	dic_self=vect.fit(corpus).get_feature_names()
	lsvc=LinearSVC(C=0.01,penalty="l2",dual=False).fit(tdm,labels)
	model=SelectFromModel(lsvc,prefit=True)
	X_select=model.transform(tdm)
	print(X_select.shape)
	index=model.get_support(indices=False)

	dic_select=[]
	n=0
	for i in index:
		if i:
			dic_select.append(dic_self[n])
		n=n+1
	return dic_select


def get_weight(x):
	if x==1:
		return 0.6
	if x in [5,6,7]:
		return 2
	else:
		return 1

# function name:
#      deploy()
# description:
#      train model and export model to file for later deploy
# train model 
def deploy():
  dic_self=get_vocabulary(60)
  if opts.rf:
	print("begin to train model\n")
#	clf=Pipeline([
#			  ('vect',CountVectorizer(min_df=1,max_df=0.5,ngram_range=(1,1),binary=False)),
#			  ('sel_f',SelectPercentile(chi2,20)),
#			  ('rf',RandomForestClassifier(30)),
#			  ])
	clf_dic=Pipeline([
			  ('vect',CountVectorizer(min_df=1,max_df=0.5,ngram_range=(1,1),binary=False,vocabulary=dic_self)),
              ('lf',LogisticRegression(penalty='l1',C=1.5) ),
#			  ('knn',KNeighborsClassifier(n_neighbors=8,weights='uniform') ),
# 			  ('rf',RandomForestClassifier(50) ),
#             ('gbdt',GradientBoostingClassifier(n_estimators=20,max_depth=5) ),
#			  ('svm',LinearSVC(C=0.8,penalty='l2',dual=False)),
			  ])

  # train model and output score for test dataset
  if opts.train:
	  
#	  t0=time()
#	  clf.fit(X_train,y_train)
#	  print("fit time: %f\n" %(time()-t0) )

#	  t0=time()
#	  pred=clf.predict(X_test)
#	  print(metrics.classification_report(y_test,pred) )
#	  print("the precision of the model is %f,cost time: %f\n" %(np.mean(pred==y_test),(time()-t0)) )
#	  print(metrics.confusion_matrix(y_test,pred) )

	  t0=time()
	  clf_dic.fit(X_train,y_train)
	  print("fit time:%f\n" %(time()-t0))

	  t0=time()
	  pred=clf_dic.predict(X_test)
	  print(metrics.classification_report(y_test,pred) )
	  print("the precision of the model_dic is %f,cost time: %f\n" %(np.mean(pred==y_test),(time()-t0)) )
	  print(metrics.confusion_matrix(y_test,pred) )

 # deploy model with all sample dataset
  if opts.deploy:
	 t0=time
#	 clf.fit(corpus,labels)
	 clf_dic.fit(corpus,labels)
#	 joblib.dump(clf,"deploy/model30/model_fr.pkl")
	 joblib.dump(clf_dic,"deploy/lr_words_bin/model_dirty.pkl")
	 print("saving model to disk...")

# run corresponding function based on user input
if __name__=='__main__':

	if opts.deploy or opts.train:
		corpus,labels=read_sample('total')
		X_train,X_test,y_train,y_test = train_test_split(corpus,labels,test_size=0.1,random_state=32)
		#X_train,y_train=corpus,labels
		#X_test,y_test=read_sample('jialiang')

		deploy()
		print("training model...OK\n")
	else:
 	 	op.print_help()
  		op.error("missing option")
  		sys.exit(1)
