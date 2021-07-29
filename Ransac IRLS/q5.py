# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 09:55:10 2021

@author: Nilesh
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy.linalg import multi_dot as dot

np.random.seed(5)
N = 33
X = np.linspace(0,10,N)
slope = 2
offset = 10
fr = 0.5
Y = slope*X + offset*np.ones(N)

outliers = np.random.normal(0,0.5,size = int(len(X)*fr))
inliers = np.random.normal(0,0.5,size = len(X) - len(outliers))
noise = np.append(outliers,inliers)
np.random.shuffle(noise)

Y_noisy1 = Y + noise

outliers = np.random.normal(0,15,size = int(len(X)*fr))
inliers = np.random.normal(0,0.5,size = len(X) - len(outliers))
noise = np.append(outliers,inliers)
np.random.shuffle(noise)
noise_ng= np.zeros(N)
Y_noisy2 = Y + abs(noise) 

fig, axarr = plt.subplots(1, 2,figsize = (10,10))
plt.sca(axarr[0])
plt.scatter(X,Y_noisy1)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(linestyle=':')
plt.legend([r"$ \sigma = 0.5 $"])
plt.title("Setup 1 - Data points")

plt.sca(axarr[1])
plt.scatter(X,Y_noisy2)
plt.xlabel("X")
#plt.ylabel("Y")
plt.grid(linestyle=':')
plt.legend([r"$ \sigma = 0.5 \: & \: \sigma = 5 $"])
plt.title("Setup 2 - Data points")

plt.show()

one = np.ones(N)
X = np.reshape(X,[N,1])
one = np.reshape(one,[N,1])
X = np.append(X,one,axis = 1)

[U,S,V] = np.linalg.svd(X)
S = np.diag(S)
solution = dot([V,np.linalg.inv(S),np.transpose(U[0:2,0:2])])
ans = dot([U[0:2,0:2],S,np.transpose(V)])

pinv = np.linalg.inv(np.dot(np.transpose(X),X))
solution = np.dot(pinv,np.transpose(X))

solution1= np.dot(solution,Y_noisy1)
yhat1 = np.dot(X,solution1)

solution2= np.dot(solution,Y_noisy2)
yhat2 = np.dot(X,solution2)

fig, axarr = plt.subplots(1, 2,figsize = (10,10))
plt.sca(axarr[0])
plt.plot(X[:,0],yhat1)
plt.scatter(X[:,0],Y_noisy1)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(linestyle=':')
plt.legend([r"$ \sigma = 0.5 $"])
plt.title("Setup 1 - OLS")

plt.sca(axarr[1])
plt.plot(X[:,0],yhat2)
plt.scatter(X[:,0],Y_noisy2)
plt.xlabel("X")
#plt.ylabel("Y")
plt.grid(linestyle=':')
plt.legend([r"$ \sigma = 0.5 \: & \: \sigma = 5 $"])
plt.title("Setup 2 - OLS")
plt.show()

w = 0.25
p = 0.99

r = int(np.log(1 - p)/np.log(1- w**2))

def getdistance(m,c,x,y):
    d = abs(y - m*x - c)/np.sqrt(1 + m**2)
    return d

def ransac(x,y,threshold):
    output = {}
    ratio = 0
    m_opt = 0
    c_opt = 0
    
    itr = 0
    
    for i in range(0,r):
        count = 0
        xIn = []
        xOut = []
        yIn = []
        yOut = []
        [j,k] = np.random.choice(range(0,N),2,replace = False)
        x0,y0,x1,y1  = x[j,0],y[j],x[k,0],y[k]
        xlist = [x0,x1]
        ylist = [y0,y1]
        m = (y1 - y0)/(x1 - x0)
        c = y0 - m*x0
        w = np.array([m,c])
        ynew = np.dot(x,np.transpose(w))
        diff = ynew - y
        for cord in range(0,N):
            d = getdistance(m,c,x[cord,0],y[cord] )
            if  d <= threshold:
                xIn.append(x[cord,0])
                yIn.append(y[cord])
                count += 1
            else:
                xOut.append(x[cord,0])
                yOut.append(y[cord])
                
        
        if count/N > ratio:
            ratio = count/N
            m_opt = m
            c_opt = c
            print(ratio)
            output[i] = [xIn,yIn,xOut,yOut,np.round(ratio,2),m,c]
            
            itr += 1
        
        
    return m_opt,c_opt,output
        
def plot_ransac(output):
    keys = sorted(output.keys())
    n = max(int(len(keys)/2),1)
    keys = keys[-2*n:]
    rows = n
    cols = 2
    fig, axarr = plt.subplots(rows, cols,figsize = (10,10))
    count = 0
    for itr in keys:
        
        [xIn,yIn,xOut,yOut,ratio,m,c] = output[itr]
        if n>1:
            plt.sca(axarr[int(count/2),count%2])
        else:
            plt.sca(axarr[count])
        plt.scatter(xIn,yIn,color = 'g')
        plt.scatter(xOut,yOut,color='r')
        plt.plot(xIn,m*np.array(xIn) + c)
        plt.title('{} iterations'.format(itr+1))
        plt.legend(["{} ratio".format(ratio)])
        count += 1
        
    fig.show()

m,c,output = ransac(X,Y_noisy2,1)
plot_ransac(output)

fig, axarr = plt.subplots(2, 2,figsize = (10,10))
plt.sca(axarr[0,0])
plt.scatter(X[:,0],Y_noisy1)
plt.plot(X[:,0],yhat1,color = 'r')
plt.title("Setup 1 - OLS")

plt.sca(axarr[0,1])
m,c,output = ransac(X,Y_noisy1,1)
plt.scatter(X[:,0],Y_noisy1)
plt.plot(X[:,0],m*X[:,0] + c,color = 'r')
plt.title("Setup 1 - RANSAC")

plt.sca(axarr[1,0])
plt.scatter(X[:,0],Y_noisy2)
plt.plot(X[:,0],yhat2,color = 'r')
plt.title("Setup 2 - OLS")

plt.sca(axarr[1,1])
m,c,output = ransac(X,Y_noisy2,1)
plt.scatter(X[:,0],Y_noisy2)
plt.plot(X[:,0],m*X[:,0] + c,color = 'r')
plt.title("Setup 2 - RANSAC")

def cost(x,y,m,c,s):
    e = abs(y-m*x-c)/(1 + m**2)
    c = e**2/(e**2 + s**2)
    return c**2

def irls(X,y,threshold,m,c):
    s = 100
    
    itrs = 10
    for i in range(0,itrs):
        phi = []
        for j in range(0,N):
            phi.append(cost(X[j,0],y[j],m,c,s))
            
        phi = np.array(phi)
        print("sum",i,sum(phi))
        phi = np.diag(phi)
        #phi = np.eye(len(phi))
        pinv = np.linalg.inv(dot([np.transpose(X),phi,X]))
        solution = dot([pinv,np.transpose(X),phi,np.transpose(y)])
        m = solution[0]
        c = solution[1]
    
    return([m,c])


[m1,c1] = irls(X,Y_noisy1,1,0,15)
[m2,c2] = irls(X,Y_noisy2,1,0,15)


fig, axarr = plt.subplots(1, 2,figsize = (10,10))
y1_irls = m1*X[:,0] + c1
y2_irls = m2*X[:,0] + c2

plt.sca(axarr[0])
plt.scatter(X[:,0],Y_noisy1)
plt.plot(X[:,0],y1_irls,color = 'g')
plt.title("Setup 1 - IRLS")

plt.sca(axarr[1])
plt.scatter(X[:,0],Y_noisy2)
plt.plot(X[:,0],y2_irls,color = 'g')
plt.title("Setup 2 - IRLS")

# Comparing IRLS initial conditions and convergence

fig, axarr = plt.subplots(2, 2,figsize = (10,10))
plt.sca(axarr[0,0])
plt.scatter(X[:,0],Y_noisy2)
[m1,c1] = irls(X,Y_noisy2,1,0,15)
plt.plot(X[:,0],m1*X[:,0] + c1,color = 'r')
plt.plot(X[:,0],0*X[:,0] + c*np.ones(len(X[:,0])),color = 'g')
plt.title("Setup 2 - IRLS")
plt.legend(["Converged","Initial"])

plt.sca(axarr[0,1])
plt.scatter(X[:,0],Y_noisy2)
[m1,c1] = irls(X,Y_noisy2,1,0,20)
plt.plot(X[:,0],m1*X[:,0] + c1,color = 'r')
plt.plot(X[:,0],0*X[:,0] + 20*np.ones(len(X[:,0])),color = 'g')
plt.title("Setup 2 - IRLS")
plt.legend(["Converged","Initial"])

plt.sca(axarr[1,0])
plt.scatter(X[:,0],Y_noisy2)
[m1,c1] = irls(X,Y_noisy2,1,0,35)
plt.plot(X[:,0],m1*X[:,0] + c1,color = 'r')
plt.plot(X[:,0],0*X[:,0] + 35*np.ones(len(X[:,0])),color = 'g')
plt.title("Setup 2 - IRLS")
plt.legend(["Converged","Initial"])

plt.sca(axarr[1,1])
plt.scatter(X[:,0],Y_noisy2)
[m1,c1] = irls(X,Y_noisy2,1,0,50)
plt.plot(X[:,0],m1*X[:,0] + c1,color = 'r')
plt.plot(X[:,0],0*X[:,0] + 50*np.ones(len(X[:,0])),color = 'g')
plt.title("Setup 2 - IRLS")
plt.legend(["Converged","Initial"])

plt.show()

# Comparing Ransac and IRLS

fig, axarr = plt.subplots(2, 2,figsize = (10,10))
plt.sca(axarr[0,0])
plt.scatter(X[:,0],Y_noisy1)
plt.plot(X[:,0],y1_irls,color = 'r')
plt.title("Setup 1 - IRLS")

plt.sca(axarr[0,1])
m,c,output = ransac(X,Y_noisy1,1)
plt.scatter(X[:,0],Y_noisy1)
plt.plot(X[:,0],m*X[:,0] + c,color = 'r')
plt.title("Setup 1 - RANSAC")

plt.sca(axarr[1,0])
plt.scatter(X[:,0],Y_noisy2)
plt.plot(X[:,0],y2_irls,color = 'r')
plt.title("Setup 2 - IRLS")

plt.sca(axarr[1,1])
m,c,output = ransac(X,Y_noisy2,1)
plt.scatter(X[:,0],Y_noisy2)
plt.plot(X[:,0],m*X[:,0] + c,color = 'r')
plt.title("Setup 2 - RANSAC")

