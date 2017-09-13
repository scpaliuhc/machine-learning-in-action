from DTree.tree import *
import matplotlib.pyplot as plt





set,labels=createDataSet()
print (chooseBestFeatureToSplit(set))
treeDic=createTree(set,labels)
print(treeDic)

set.labels=createDataSet()#因为在创建Tree时就已经修改了labels(del(……))本来创建树和使用数的数据集就应该是两套
print(classify(treeDic,labels,[1,0]))

