#coding:utf-8
from KNN.knn import *
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
#
# mat,labels=k.file2matrix("datingTestSet.txt")
# datingMat=k.autoNorm(mat)
# print(labels)
# fig=plt.figure("dating")
# ax1=fig.add_subplot(111)
# ax1.scatter(mat[:,1],mat[:,2],20.0*array(labels),20.0*array(labels))
# plt.show()

'''
test error rate
'''
# def datingClassTest_noNorm(k):
#     ratio=0.1
#     datingDataMat,datingLabels=file2matrix("datingTestSet.txt")
#     m=datingDataMat.shape[0]
#     numTestVecs=int(m*ratio)
#     errorcount=0
#     for i in range(numTestVecs):
#         classifierResult=classify0(datingDataMat[i,:],datingDataMat[numTestVecs:,:],datingLabels[numTestVecs:],k)
#         print('the classifier came back with %d, the real answer is %d'%(classifierResult,datingLabels[i]))
#         if classifierResult!=datingLabels[i]:
#             errorcount+=1
#     print("the total error rate is %f"%(errorcount/float(numTestVecs)))
#
# def datingClassTest_norm(k):
#     ratio=0.1
#     datingDataMat,datingLabels=file2matrix("datingTestSet.txt")
#     normMat,ranges,minVals=autoNorm(datingDataMat)
#     m=normMat.shape[0]
#     numTestVecs=int(m*ratio)
#     errorcount=0
#     for i in range(numTestVecs):
#         classifierResult=classify0(normMat[i,:],normMat[numTestVecs:,:],datingLabels[numTestVecs:],k)
#         print('the classifier came back with %d, the real answer is %d'%(classifierResult,datingLabels[i]))
#         if classifierResult!=datingLabels[i]:
#             errorcount+=1
#     print("the total error rate is %f"%(errorcount/float(numTestVecs)))
# datingClassTest_noNorm(5)
# datingClassTest_norm(5)

'''
手写数字
'''
testVector=img2vector('testDigits/0_13.txt')
print(testVector)



