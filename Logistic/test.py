from Logistic.logRegres import *
import threading
dataMat,labelMat=loadDataSet()
weights1=stocGradAscent(dataMat,labelMat)
print(weights1)
weights2=gradAscent(dataMat,labelMat)
print(weights2)
plotBestFit(weights1)
