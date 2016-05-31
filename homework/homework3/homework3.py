#11621003-csmath3#
import numpy
from matplotlib import pyplot as plt
d = 2
SampleNum = 1000
ClassNum = 2

sigma = []
mu = []
L = []

for i in range(ClassNum):
    muTmp = numpy.random.uniform(-1,1,d) + 3*i
    mu.append(muTmp)
    LTmp = numpy.random.uniform(-1,1,(d, d))
    LTmp = numpy.mat(LTmp)
    LTmp = LTmp + numpy.identity(d)
    while(abs(numpy.linalg.det(LTmp)) < 1e-10):
        LTmp = LTmp + numpy.identity(d)
    L.append(LTmp)
    sigma.append(LTmp*LTmp.T)

pi = numpy.random.uniform(0.33, 0.67, ClassNum-1)
pi.sort
pi.resize(ClassNum)
pi[ClassNum-1] = 1

data = []
label = []

for i in range(SampleNum):
    T = 0
    z = numpy.random.uniform(0,1)
    while(z>pi[T]):
        T = T + 1
    muSample = mu[T]
    LSample = L[T]
    dataTmp = numpy.random.normal(0,1,(d,1))
    dataTmp = numpy.matrix(dataTmp)
    dataTmp = LSample*dataTmp + numpy.mat(muSample).T
    data.append(dataTmp)
    label.append(T)


plt.scatter([i[0,0] for i in data], [i[1,0] for i in data],c=label,cmap='cool')
plt.scatter([i[0] for i in mu], [i[1] for i in mu])

mu0 = mu
muReg0 = numpy.zeros((d,1))
sigmaReg0 = numpy.identity(d)
sigma0 = sigma
pi0 = numpy.zeros((ClassNum,1))
pi0[0] = pi[0]
for i in range(ClassNum):
    mu0[i] = numpy.mat(mu0[i]).T + 5*numpy.ones((d,1))
    sigma0[i] = sigma0[i] + 5*numpy.identity(d)
for i in range(1,ClassNum):
    pi0[i] = pi[i] - pi[i-1]
mu0Reg = mu0
sigma0Reg = sigma0

T = 0
r = numpy.zeros((SampleNum, ClassNum))
N = numpy.zeros(ClassNum)

beta = 1
niu = 2

while(T<10):
    T = T + 1
    for k in range(SampleNum):
        s = 0
        for j in range(ClassNum):
            s = s + pi0[j]/((numpy.linalg.det(sigma0[j]))**(d/2))*numpy.exp(-(data[:][k] - mu0[j]).T*sigma0[j].I/2*(data[:][k] - mu0[j]))
        for j in range(ClassNum):
            r[k,j] = pi0[j] /(s * (numpy.linalg.det(sigma0[j]))**(d/2))*numpy.exp(-(data[:][k] - mu0[j]).T*sigma0[j].I/2*(data[:][k] - mu0[j]))
            N[j] = sum(r[:,j])
            pi0[j] = 1.0*N[j] / SampleNum
    for j in range(ClassNum):
        tmpMu = numpy.zeros((d,1))
        for i in range(d):
            tmpMu[i] = 1.0 / mu0[j][i]
        mu0[j] = numpy.zeros((d,1))
        mu0Reg[j] = numpy.zeros((d,1))
        for k in range(SampleNum):
            mu0[j] = mu0[j] + r[k,j] * data[:][k]
        mu0[j] = mu0[j] / N[j]

    for j in range(ClassNum):
        s = numpy.zeros((d,d))
        for k in range(SampleNum):
            s = s + r[k,j]*(data[:][k] - mu0[j])*(data[:][k] - mu0[j]).T
	sigma0[j] = (s+numpy.identity(d))/(N[j]-1)
        print numpy.linalg.det(sigma0[j])



eig1 = numpy.linalg.eig(sigma0[0])
eig2 = numpy.linalg.eig(sigma0[1])
sigma1 = numpy.mat(numpy.identity(2))
sigma2 = numpy.mat(numpy.identity(2))
sigma1[0,0] = numpy.sqrt(eig1[0][0])
sigma1[1,1,] = numpy.sqrt(eig1[0][1])
sigma2[0,0] = numpy.sqrt(eig2[0][0])
sigma2[1,1] = numpy.sqrt(eig2[0][1])
sigma1 = numpy.mat(eig1[1])*sigma1*numpy.mat(eig1[1]).T
sigma2 = numpy.mat(eig2[1])*sigma2*numpy.mat(eig2[1]).T

theta = [2*numpy.pi*i/100 for i in range(100)]
x = numpy.mat(numpy.zeros((2,100)))
y1 = numpy.mat(numpy.zeros((2,100)))
y2 = numpy.mat(numpy.zeros((2,100)))
for i in range(100):
    x[0,i] = numpy.cos(theta[i])
    x[1,i] = numpy.sin(theta[i])
    y1[:,i] = sigma1*x[:,i]+mu0[0]
    y2[:,i] = sigma2*x[:,i]+mu0[1]

y11 = []
y12 = []
y21 = []
y22 = []
for i in range(100):
    y11.append(y1[0,i])
    y12.append(y1[1,i])
    y21.append(y2[0,i])
    y22.append(y2[1,i])
plt.show()
plt.scatter([i[0,0] for i in data], [i[1,0] for i in data],c=label,cmap='cool')
plt.scatter([i[0,0] for i in mu0], [i[1,0] for i in mu0])
plt.scatter(y11,y12)
plt.scatter(y21,y22)

plt.show()

