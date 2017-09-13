#coding:utf-8


from numpy import *
'''
手写数字
'''
def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        line=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(line[j])
    return returnVect

'''
约会预测
'''
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numOfLines=len(arrayOLines)
    returnMat=zeros((numOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFormLine=line.split('\t')
        returnMat[index,:]=listFormLine[0:3]

        if listFormLine[-1]=='largeDoses':
            classLabelVector.append(0)

        elif listFormLine[-1]=='smallDoses':
            classLabelVector.append(1)

        elif listFormLine[-1]=='didntLike':
            classLabelVector.append(2)
        index += 1
    fr.close()
    return returnMat,classLabelVector
def autoNorm(dataset):
    minVals=dataset.min(0)#minVals.shape=(1*3)
    maxVals=dataset.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataset))
    m=dataset.shape[0]
    normDataSet=dataset-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals
'''
原始的方法
'''
def createDataset():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def dict2list(dic):
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst

def classify0(inX,dataset,labels,k):
    dataSetSize=dataset.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataset
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(dict2list(classCount),key=lambda d:d[1],reverse=True)
    return sortedClassCount[0][0]

