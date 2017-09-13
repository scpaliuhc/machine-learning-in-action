#coding:utf-8

def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):#产生初始候选集 每个候选集是frozenSet对象
    C1=[]
    for t in dataSet:
        for item in t:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # print(C1)
    return list(map(frozenset,C1))#map(function, iterable, ...)对iterable中每个元素使用function方法 这里是对C1中的每一项构建了一个不变集合
                             #目的是要使用这些集合作为字典键值，set是不可以作为dict的键值的

def scanD(D,Ck,minSupport):#产生Lk频繁项集
    ssCnt={}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt: ssCnt[can]=1#集合作为key值 python2是has_key python3改为了in
                else: ssCnt[can]+=1

    numItems = float(len(D))
    retList=[]
    supportData={}
    for key in ssCnt:
        support=ssCnt[key]/numItems
        if support>=minSupport:
            retList.insert(0,key)
        supportData[key]=support
    return retList,supportData#Lk是list，但中间每个元素是frozenset

def aprioriGen(Lk,k):#产生Ck候选集
    if(len(Lk)==1):
        print("End searching!")
        return []
    retList=[]
    lenLk=len(Lk)
    end=k-2
    for i in range(lenLk):
        L1 = list(Lk[i])[:end]
        L1.sort()
        for j in range(i+1,lenLk):
            L2=list(Lk[j])[:end]
            L2.sort()
            if L1==L2:
                retList.append(frozenset(Lk[i]|Lk[j]))
    return retList

def apriori(dataSet,miniSupport=0.5):
    C1=createC1(dataSet)#产生初始候选集
    D=list(map(set,dataSet))#将每条交易变成集合(方便使用issubset函数)
    L1,supportData=scanD(D,C1,miniSupport)#产生一阶频繁项集和候选集的支持度
    L=[L1]#记录所有阶的频繁项集
    k=2
    while(len(L[k-2])>1):#产生之后的频繁项集，当某一阶的频繁项集只有一个时就可以结束搜索
        Ck=aprioriGen(L[k-2],k)#产生k阶候选集
        Lk,suppk=scanD(D,Ck,miniSupport)#产生k阶频繁集和所有k阶候选集的支持度
        L.append(Lk)
        supportData.update(suppk)
        k+=1
    return L,supportData

def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=0.7):#H记录规则右边可能出现的情况
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence   supportData[freqSet-conseq]：从集合中移除一个元素
        if conf >= minConf:
            print ('rule:',freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])#每个平凡集中有几个元素
    # if (len(freqSet) > (m + 1)): #因为当前右部有m个元素，下一步合成m+1个元素，如果频繁项集中的个数支持下一步的合并才合并
    #     Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
    #     Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
    #     if (len(Hmp1) > 1):    #need at least two sets to merge
    #         rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
    while(len(freqSet)>m):
        H=calcConf(freqSet,H,supportData,brl,minConf)
        if(len(H)>1):#进一步合并
            H=aprioriGen(H,m+1)
            m+=1
        else:
            break




