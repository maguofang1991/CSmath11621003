# -*- coding: utf-8 -*-
#11621003-cshomework2#

import numpy as np
import pylab as pl


def pca(X, topNfeat=64):
    meanVals = np.mean(X, axis=0)
    DataAdjust = X - meanVals
    covMat = np.cov(DataAdjust, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat)) #特征值和特征向量
    eigValInd = np.argsort(eigVals)            #排序
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:, eigValInd]       #特征向量
    lowData = X * redEigVects      #数据转换到低维新空间
    reconMat = (lowData * redEigVects.T) + meanVals
    return lowData, reconMat, meanVals

def loadDataSet(fileName, delim=','):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return np.mat(datArr)




dataMat = loadDataSet('data/train.txt')
m = 8
lowData, reconMat, immean = pca(dataMat)
fig = pl.figure()
pl.gray()
pl.imshow(immean.reshape(m, m))
pl.show()

fig = pl.figure()
ax = fig.add_subplot(111)
ax.scatter(lowData[:, 0], lowData[:, 1], marker='s', s=5)

X = [-20, -10, 0, 10, 20]
Y = [-20, -10, 0, 10, 20]
plotX = [99 for n in range(25)]
plotY = [99 for n in range(25)]
plotfig = [0 for n in range(25)]
count = 0
for lowDData in lowData:
    for i in range(25):
        x = i % 5
        y = i / 5
        dist_raw = abs(plotX[i] - X[x])*abs(plotX[i] - X[x])+abs(plotY[i] - Y[y])*abs(plotY[i] - Y[y])

        dist = abs(lowDData[0, 0] - X[x])*abs(lowDData[0, 0] - X[x])+abs(lowDData[0, 1] - Y[y])*abs(lowDData[0, 1] - Y[y])

        if dist < dist_raw:
            plotX[i] = lowDData[0, 0]
            plotY[i] = lowDData[0, 1]
            plotfig[i] = reconMat[count]
    count = count + 1

ax.scatter(plotX[:], plotY[:], c='blue',marker='o', s=50)
pl.show()
pl.subplot(5, 5, 1)
for i in range(25):
    pl.subplot(5, 5, i)
    pl.imshow(plotfig[i].reshape(8, 8))
pl.show()
