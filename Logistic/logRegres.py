#coding:utf-8
'''
随机梯度比传统的梯度法收敛更快，波动也较少
'''
from numpy import *
def loadDataSet():
    dataMat=[];labelMat=[]
    fr=open("testSet.txt")
    for line in fr.readlines():
        lineArr=line.strip().split()#strip()默认移除头尾的空格
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(x):
    return 1.0/(1+exp(-x))#用的是numpy的exp函数

def gradAscent(dataMat,labelMat):
    dataMatrix=mat(dataMat)
    labelMatrix=mat(labelMat).transpose()

    m,n=shape(dataMatrix)
    alpha=0.01
    maxCycles=500
    weights=ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)#所有样本
        error=labelMatrix-h
        weights=weights+alpha*dataMatrix.transpose()*error##????
    return weights

def stocGradAscent(dataMat,labelMat,numIter=150):
    m,n=shape(dataMat)
    dataArr=array(dataMat)#dataMat是list类型，想要进行后续计算需要使用ndarray类型
    weights=ones(n)
    for j in range(numIter):
        dataIndex=range(m)
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataArr[randIndex] * weights))
            error = float(labelMat[randIndex] - h)
            weights = weights + alpha * error * dataArr[randIndex]

    return weights
def plotBestFit(wei):
    import matplotlib.pyplot as plt
    try:
        weights=wei.getA()#从矩阵转为ndarray
    except:
        weights=wei
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i, 1]);ycord2.append(dataArr[i, 2])

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')#画
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]#w0+w1x+w2y=0
    ax.plot(x,y)
    plt.xlabel('x1');plt.ylabel('x2')
    plt.show()
