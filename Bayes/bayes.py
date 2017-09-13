'''
Created on Oct 19, 2010

@author: Peter
'''

import codecs
from numpy import *
#产生训练数据
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,0,0,1,1,1]    #1 is abusive, 0 not
    return postingList,classVec

#得到单词集合
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

#将输入转为数字向量
def setOfWords2Vec(vocabList, inputSet):#词集模型，只关注是否出现
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

#得到需要的概率
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)#有多少文档
    numWords = len(trainMatrix[0])#一共有多少单词
    pAbusive = sum(trainCategory)/float(numTrainDocs)#正样本的概率
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones(),防止某个P(wi|c)=0使得P(c|w)=0
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0,为什么是2.0
    for i in range(numTrainDocs):#i代表各个文档中词条出现情况
        if trainCategory[i] == 1:#若是正样本
            p1Num += trainMatrix[i]#若某个单词出现在i文档中，则对应的记录加1
            p1Denom += sum(trainMatrix[i])#i文档中，共有多少种特殊单词出现，累加
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)         #change to log()
    return p0Vect,p1Vect,pAbusive

#对已经转为数字向量的输入进行分类
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult,vec2Classify * p1Vec:对应元素相乘，即待测试文档各个单词的概率
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#
def bagOfWords2VecMN(vocabList, inputSet):#词袋模型，关注了出现的次数
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
#
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(bagOfWords2VecMN(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(bagOfWords2VecMN(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(bagOfWords2VecMN(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

#切分文本
def textParse(bigString):    #input is big string, #output is word list
    import re#正则表达式
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
#
def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(codecs.open('email/spam/%d.txt' % i,"r","gbk").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        try:
            wordList = textParse(codecs.open('email/ham/%d.txt' % i,"r","gbk").read())
        except:
            print(i)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary所有的单词列表
    #50个集合，从中随机选10个作为测试集
    trainingSet = list(range(50)); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    #构建训练集的数字向量集合
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    #从训练集中得到所需的向量
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error",docList[docIndex])
    print('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText
#
# def calcMostFreq(vocabList,fullText):
#     import operator
#     freqDict = {}
#     for token in vocabList:
#         freqDict[token]=fullText.count(token)
#     sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
#     return sortedFreq[:30]
#
# def localWords(feed1,feed0):
#     import feedparser
#     docList=[]; classList = []; fullText =[]
#     minLen = min(len(feed1['entries']),len(feed0['entries']))
#     for i in range(minLen):
#         wordList = textParse(feed1['entries'][i]['summary'])
#         docList.append(wordList)
#         fullText.extend(wordList)
#         classList.append(1) #NY is class 1
#         wordList = textParse(feed0['entries'][i]['summary'])
#         docList.append(wordList)
#         fullText.extend(wordList)
#         classList.append(0)
#     vocabList = createVocabList(docList)#create vocabulary
#     top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
#     for pairW in top30Words:
#         if pairW[0] in vocabList: vocabList.remove(pairW[0])
#     trainingSet = range(2*minLen); testSet=[]           #create test set
#     for i in range(20):
#         randIndex = int(random.uniform(0,len(trainingSet)))
#         testSet.append(trainingSet[randIndex])
#         del(trainingSet[randIndex])
#     trainMat=[]; trainClasses = []
#     for docIndex in trainingSet:#train the classifier (get probs) trainNB0
#         trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
#         trainClasses.append(classList[docIndex])
#     p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
#     errorCount = 0
#     for docIndex in testSet:        #classify the remaining items
#         wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
#         if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
#             errorCount += 1
#     print 'the error rate is: ',float(errorCount)/len(testSet)
#     return vocabList,p0V,p1V
#
# def getTopWords(ny,sf):
#     import operator
#     vocabList,p0V,p1V=localWords(ny,sf)
#     topNY=[]; topSF=[]
#     for i in range(len(p0V)):
#         if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
#         if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
#     sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
#     print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
#     for item in sortedSF:
#         print item[0]
#     sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
#     print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
#     for item in sortedNY:
#         print item[0]
