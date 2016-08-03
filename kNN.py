#-*- coding=utf-8 -*-
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1))-dataSet  #计算待预测数据与每个样本向量的差
    sqDiffMat = diffMat**2                         
    sqDistance = sqDiffMat.sum(axis=1)            #计算距离的平方
    distance = sqDistance**0.5
    sortedDistIndicines = distance.argsort()      #将距离按递增的顺序排列，并返回index
    classCount={}
    for i in range(k):                            #统计前k个样本所属的类别，找出最多类别
        voteIlabel = labels[sortedDistIndicines[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),
     key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#data,label=createDataSet()
#test = array([0,0])
#result = classify0(test,data,label,2)
#print(result)

#将文本记录转化为NumPy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()                 #读取文件，返回一个列表，每一行为列表的一个元素
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))         #生成一个numberOfLines*3的零阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()                      #清除字符串中开头或结尾的空格
        listFromLine = line.split('\t')          #以'\t'为分隔标识分隔字符串
        returnMat[index,:] = listFromLine[0:3]   #截取前3个元素
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape(0)                          #m为行数
    normDataSet = dataSet - tile(minVals, (m,1))  #tile() 重构矩阵
    normDataSet = normDataSet/tile(ranges,(m,1))  #归一化
    return normDataSet, ranges, minVals
