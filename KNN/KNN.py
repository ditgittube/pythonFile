#-*- coding=utf-8 -*-
from numpy import *
import operator
from os import listdir

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

#测试分类器
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)                                              #numTestVecs为测试样本数目
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m, :],   
                                     datingLabels[numTestVecs:m], 3)          #获取每个测试样本的分类结果
        print("the classifier came back with: %d, the real answer is: %d"\
              % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):                             #计算错误率
            errorCount += 1
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    
#约会网站测试数据
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input("percentage of time spent playing vedio games?")) #收集三个问题的答案
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels,3)
    print("You will probably like this person:",resultList[classifierResult - 1])   #实际类别转化为列表下标
    
#手写识别系统
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')                        #导入文件夹得文件名
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLables.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnder,\
                                     trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d "\
               % (classifierResult, classNumstr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))
              