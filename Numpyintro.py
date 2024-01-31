# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:06:07 2024

@author: bongw
"""

import numpy as np

#conversional pythos for loop
print("using just python:")
for i in range(1,11):
    print(i)
    
#numpy - arange
print("using numpy:")
for i in np.arange(1,11,0.5):
        print(i)
        
#suqaring the number from 1-5
squares=[]
for i in range(1,6):
    squares.append(i**2)
print(squares)

sqaures = np.arange(1,6)**2
print(sqaures)
    
import matplotlib.pyplot as plt
x = np.arange(1,6)
y = x**2
print("shape of x:")
print(x.shape)
print("shape of y:")
print(y.shape)
print(x+y)
plt.plot(x,y,"r*")
plt.show()

x = np.arange(1,6)
y = x**2
print("shape of x:")
print(x.shape)
print("shape of y:")
print(y.shape)
print(x*y)
print("calculating dot product")
print(x.dot(y))
print("calculate cross product")
print(np.matmul(x,y))
plt.plot(x,y,"r*")
plt.show()


alist = [1,2,5,6,15,22]
data = np.array(alist)
print(data)
data2 = data.reshape([2,3])
data3 = data.reshape([2,3])
print("data 2")
print(data2)
print("data 3")
print(data3)

alist2=[[ 1,2,5], [6,15,22]]

#multiply two 2X3 matrices
data4=np.matmul(data2.T,data3)
print('data4:')
print(data4)

#Cross product of matrices
print('cross product')
crossdata=np.matmul(data2[0,:], data3[1,:])
print(crossdata)

a = np.array([[1,2,3],[4,5,6],[7,8,-9]])
b = np.array([3,-4,2])
d = np.linalg.det(a)
if(d>0):
    print(f"d = {d},so this matrix is solvable")
sol = np.linalg.solve(a,b)

#Working ith the soisy data
data = np.loadtxt("noisydata.csv", skiprows=1, delimiter=",")
data_avg=np.mean(data,0)
print(data_avg)
pressure=data[:,0]
flowrate=data[:,1]

#To show/count the pressure data points, for example, those smaller than 40. Use the ff


fit=np.polyfit(pressure,flowrate,3)
flowfit=np.polyval(fit,pressure)
plt.plot(pressure,flowrate,"go")
plt.xlabel("pressure (Pa)")
plt.ylabel("flow rate ($m^3/s$")
plt.plot(pressure,flowrate,"k-")
plt.title("chemical experiment")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
n=5000
x = np.random.uniform (size=n)
y = np.random.uniform(size=n)
print(sum(x*x+y*y <=1)/n*4)
plt.plot(x[x*x+y*y<=1] ,y [x*x+y*y<=1], "bo" )
plt.plot(x[x*x+y*y>1], y [x*x+y*y>1], "ro")
plt.title("calculating $\pi$")
plt.savefig("pi.jpg") #To save your picture as a jpg file
plt.show()







