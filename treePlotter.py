# -*- encoding: utf-8 -*-
#使用文本绘制树节点
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,\
                            xycoords='axes fraction',\
                            xytext=centerPt, textcoords='axes fraction',\
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
    
def createPlot():
    fig = plt.figure(1, facecoler='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, fraeon=False)
    plotNode(U'决策节点', (0.5, 0.1),(0.1, 0.5),decisionNode)
    plotNode(U'叶节点', (0.8, 0.1),(0.3, 0.8), leafNode)
    plt.show()