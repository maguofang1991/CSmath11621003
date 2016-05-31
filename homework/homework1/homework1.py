# -*- coding: utf-8 -*-
#11621003-csmath-homework1#
import pylab as pl
import numpy as np
from scipy.optimize import leastsq

def real_function(x):
    return np.sin(2*np.pi*x)

def fake_function(p, x):
    f = np.poly1d(p)
    return f(x)

def residuals(p, y, x):
    return y - fake_function(p, x)


def regularization_residuals(p, y, x):
    ret = y - fake_function(p, x)
    ret = np.append(ret, np.sqrt(lambdar)*p)
    return ret

def samply(n):
    x = np.linspace(0, 1, n)
    x_show = np.linspace(0, 1, 1000)
    y0 = real_function(x)
    y1 = [np.random.normal(0, 0.1) + y for y in y0]
    return x, y1, x_show

def compute_line(x, y, order, r):
    p = []
    for xi in x:
        pi = []
        for j in range(0, order+1):
            pi.append(np.power(xi, j))
        p.append(pi)
    P = np.array(p)
    Y = np.array([y])


    W = np.dot(np.dot(np.linalg.inv(np.dot(P.T, P)+np.eye(order+1, order+1)*r), P.T), Y.T)
    print W
    x2 = np.arange(-np.pi-1, np.pi+1, 0.01)
    y2 = []
    for xi in x2:
        sum = 0
        for j in range(0 , order+1):
            sum += np.power(xi, j)*W[j][0]
        y2.append(sum)

    return x2, y2

def figure(x, x_show, y, plsq):
    pl.plot(x_show, real_function(x_show), color="blue", label='real')
    pl.plot(x_show, fake_function(plsq[0], x_show), color="red", label='fitted curve')
    pl.plot(x, y, 'bo',color="black" )
    pl.legend()
    pl.show()


#sample the function curve of y=sin(x) with Gaussian noise
m = 3
n = 10
x, y, x_show = samply(n)
p0 = np.random.randn(m+1)
plsq = leastsq(residuals, p0, args=(y, x))
print '1.Fitting Parameters ：', plsq[0]
figure(x, x_show, y, plsq)

# fit degree 3 and 9 curves in 10 samples
m = 9
p0 = np.random.randn(m+1)
plsq = leastsq(residuals, p0, args=(y, x))
print '2.Fitting Parameters ：', plsq[0]
figure(x, x_show, y, plsq)


# fit degree 9 curves in 15 samples
n=15
x, y, x_show = samply(n)
p0 = np.random.randn(m+1)
plsq = leastsq(residuals, p0, args=(y, x))
print '3.1.Fitting Parameters ：', plsq[0]
figure(x, x_show, y, plsq)


# fit degree 9 curves in 100 samples
n=100
x, y, x_show = samply(n)
p0 = np.random.randn(m+1)
plsq = leastsq(residuals, p0, args=(y, x))
print '3.2.Fitting Parameters ：', plsq[0]
figure(x, x_show, y, plsq)

#fit degree 9 curve in 10 samples but with regularization term
lambdar=0.0001
p0 = np.random.randn(m+1)
plsq = leastsq(regularization_residuals, p0, args=(y, x))
print '4.Fitting Parameters ：', plsq[0]
figure(x, x_show, y, plsq)
