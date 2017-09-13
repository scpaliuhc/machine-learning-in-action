from Apriori.apriori import *
'''
map()对象后要用list才可以使用其对象
set与frozenset
'''
dataset=loadDataSet()
L,S=apriori(dataset)
rules=generateRules(L,S)
print(rules)
