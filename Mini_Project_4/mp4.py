import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invwishart, invgamma

np.random.seed(1)

#intial values


cov=np.array([[1,0],[0,2]])
d0=np.array([[4,0],[0,5]])
d=2
v0=5
N=100
y=np.random.multivariate_normal([0,0],cov,N)


# Q1 MLE 
s=np.dot(y.T,y)
print("MLE :")
print(s/N)
# Q2 Bayesian estimation

print("Bayesian estimation :")

cov_est=np.add(d0,s)/(v0+N+d+1)
print(cov_est)

#Q3 Non informative prior

# Priors that you get by setting your delta0 paramter as zero
j1=4
j2=3
print("Jeffrey's prior")
cov_est=s/(j1+N)
print(cov_est)

print("Independent Jeffrey's prior")
cov_est=s/(j2+N)
print(cov_est)

# Q4 Monte carlo estimation

M=[1000,10000,100000]

print("Monte Carlo Bayesian Estimation")

for m in M:
	print("m =",m)
	x=invwishart.rvs(v0,d0,size=m)

	lx=invwishart.pdf(x.T, df=N-3, scale=s)

	A=np.dot(x.T,lx)/np.sum(lx)
	print(A)

d0=np.array([[2,0],[0,4]])
print(" part b")
for m in M:
	print("m =",m)
	x=invwishart.rvs(v0,d0,size=m)

	lx=invwishart.pdf(x.T, df=N-3, scale=s)

	A=np.dot(x.T,lx)/np.sum(lx)
	print(A)

