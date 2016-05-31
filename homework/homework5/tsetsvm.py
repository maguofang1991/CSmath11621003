#11621003-CSMATH-SVM
  
from numpy import *  
import SVM 

print "----s1: loading----"
dataSet = []  
labels = []  
fileIn = open('test.txt')
for line in fileIn.readlines():  
    lineArr = line.strip().split('\t')  
    dataSet.append([float(lineArr[0]), float(lineArr[1])])  
    labels.append(float(lineArr[2]))  
  
dataSet = mat(dataSet)  
labels = mat(labels).T  
train_x = dataSet[0:81, :]  
train_y = labels[0:81, :]  
test_x = dataSet[80:101, :]  
test_y = labels[80:101, :]  

print "----s2: training----"
C = 0.6  
toler = 0.001  
maxIter = 50  
svmClassifier = SVM.trainSVM(train_x, train_y, C, toler, maxIter, kOption = ('linear', 0))
  

print "----s3: testing----"
accuracy = SVM.testSVM(svmClassifier, test_x, test_y)  

print "----s4: resulting----"
print 'accuracy: %.3f%%' % (accuracy * 100)
SVM.showSVM(svmClassifier)
