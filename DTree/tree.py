#coding:utf-8

from math import log

'''
下面用到的构造决策树的算法是ID3
'''
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:#返回的是字典中的key
        pro=float(labelCounts[key])/numEntries
        shannonEnt-=pro*log(pro,2)
    return shannonEnt

def createDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

def splitDataSet(dataSet,axix,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axix]==value:
            #构建新的记录
            reducedFeatVec=featVec[:axix]
            reducedFeatVec.extend(featVec[axix+1:])#扩展列表
            #添加到retDataSet
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numOfData=len(dataSet)
    numFeatures=len(dataSet[0])-1#因为记录的最后一个是标签不是特征，灵活应用
    baseEntropy=calcShannonEnt(dataSet)#整个数据集的熵
    bestInfoGain=0.0;bestFeature=-1
    for i in range(numFeatures):#按特征索引遍历
        featList=[example[i] for example in dataSet]#将每个记录中对应位置的特征取出，组成新的列表
        uniqueVals=set(featList)#得到列表中所有不重复的值，集合数据类型
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(numOfData)
            newEntropy+=prob*calcShannonEnt(subDataSet)#先算某一个特征以及其某一个取值的熵,再乘以其概率，然后通过循环结构累加
        infoGain=baseEntropy-newEntropy#计算信息增益的公式
        if infoGain>bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

def dic2list(dic):
    keys=dic.keys()
    values=dic.values()
    lis=[(key,value) for key,value in zip(keys,values)]
    return lis

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(dic2list(classCount),lambda d:d[0],reversed=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):#labels是各个特征的名称列表

    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):#classList.count(classList[0]):classList中值为classList[0]的个数
        return classList[0]#classList中的类别完全相同，返回此类别
    if len(dataSet[0])==1:#遍历完所有特征，于是按照多数表决的方法返回classList中大多数的类别
        return majorityCnt(classList)

    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}#节点的表示！！！

    del(labels[bestFeat])#删除变量labels[bestFeat]，以及解除labels[bestFeat]对其指向的值的引用
                         #因为以后不会再用到这个特征，所以需要将其从列表中移除，保证chooseBestFeatureToSplit(dataSet)返回的下标与labels中的特征对应
    featValues=[example[bestFeat] for example in dataSet]#获得所有记录在bestFeat上的取值
    uniqueVals=set(featValues)#获得bestFeat所有可能取值的集合
    for value in uniqueVals:
        subLabels=labels[:]#为什么要拷贝一份？？？
                           #不拷贝也可以实现功能，但是就改变了原列表的类容，到最后不能追溯每个节点对应的其他特征
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)

    return myTree

def classify(inputTree,featLabels,testVec):
    firstFeat=list(inputTree.keys())[0]
    secondDict=inputTree[firstFeat]
    featIndex=featLabels.index(firstFeat)#获得列表中某个元素的索引
    for key in secondDict:#特征值
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classify(secondDict[key],featLabels,testVec)
            else:
                global classLabel
                classLabel=secondDict[key]
    return classLabel

def sotreTree(inputTree,fileName):
    import pickle
    fw=open(fileName,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(fileName):
    import pickle
    fr=open(fileName,'r')
    return pickle.load(fr)


