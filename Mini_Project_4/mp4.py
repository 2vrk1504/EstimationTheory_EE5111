import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

#intial values


cov=np.array([[1,0],[0,2]])
d0=np.array([[4,0],[0,5]])
d=2
v0=5
N=100
y=np.random.multivariate_normal([0,0],cov,N)


# Q1 MLE 
cov_est=np.dot(y.T,y)/N
print("MLE :")
print(cov_est)
# Q2 Bayesian estimation

print("Bayesian estimation :")
s=np.zeros((2,2))
for i in range(N):
	a=y[i].reshape((2,1))
	s=np.add(s,np.dot(a,a.T))
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


