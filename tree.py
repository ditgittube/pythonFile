# -*- encoding: utf-8 -*-

from math import log
import operator

#计算香农信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:                   #计算每个类别的数目
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:                   #计算香农信息熵
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#生成数据集
def createDataSet():
    dataset = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataset,labels

#按给定特征划分数据集
def splitDataSet(dataSet, axis ,value):
    retDataSet = []
    for featVec in dataSet:                   #按划分位置抽取数据,axis为分类属性
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0])-1
    baseEnriopy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeature):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)                       #set()去重
        newEntropy = 0.0
        for value in uniqueVals:                         #计算划分后的信息熵newEntropy
            subDataSet = splitDataSet(dataSet,i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEnriopy - newEntropy              #保留划分后增益最大(熵减最多)，并返回对应的划分属性
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#抽出出现频率最高的分类名称
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount += 1
    sortedClassCount = sorted(classCount.iteritems(),\
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):     #如果类别相同则停止分类
        return classList[0]
    if len(dataSet[0]) == 1:                                #已遍历完所有特征时，返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])                                    #将本层最优属性删除后作为
    featValues = [example[bestFeat] for example in dataSet]  #得到列表中最优属性的属性值
    uniqueVals = set(featValues)
    for value in uniqueVals:                                 #递归地求解决策树
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet\
                                                  (dataSet,bestFeat,value),subLabels)
    return myTree