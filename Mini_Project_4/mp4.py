import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invwishart, invgamma

np.random.seed(1)

#intial values
cov=np.array([[1,0],[0,2]])
d0=np.array([[4,0],[0,5]])
d=2		# dimensions
v0=5
N=1000
y = np.random.multivariate_normal([0,0],cov,N)

print("N: ", N)
# Q1 MLE 
s=np.dot(y.T,y)
print("MLE :")
print(s/N)

# Q2 Bayesian estimation
print("Bayesian estimation :")

cov_est= (d0+s)/(v0+N+d+1)
print(cov_est)

# Q3 Non informative prior
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
	sigmas = invwishart.rvs(v0,d0,size=m)
	lx = invwishart.pdf(sigmas.T, df=N-3, scale=s)

	A=np.dot(sigmas.T,lx)/np.sum(lx)
	print(A)

d0=np.array([[2,0],[0,4]])
print(" part b")
for m in M:
	print("m =",m)
	sigmas = invwishart.rvs(v0,d0,size=m)
	lx = invwishart.pdf(sigmas.T, df=N-3, scale=s)
	A = np.dot(sigmas.T,lx)/np.sum(lx)
	print(A)

# Q5 Gibbs Sampling
print("Gibbs sampler")
A1 = A2 = 0.05
M = 1000

a1 = A1; a2 = A2;
for m in range(M):
	scale_mat = np.diag([1/a1, 1/a2])*2*v0 + s
	sigma_est5 = invwishart.rvs(df=v0+d+N-1, scale=scale_mat)
	sig_inv = np.linalg.inv(sigma_est5)

	a1 = invgamma.rvs((v0+N)/2, scale=(v0*sig_inv[0][0] + 1/A1/A1))
	a2 = invgamma.rvs((v0+N)/2, scale=(v0*sig_inv[1][1] + 1/A2/A2))
print(sigma_est5) 

# Q6 Empirical Bayes
print("Empirical Bayes")
eps = 1000
thresh = 0.0001
v_opt = N-1
step = 0
while eps >= thresh:
	step += 1
	print("Step: {}".format(step), end="\r")
	tot = np.array([1/(v_opt+N-1-i-j+1e-10) for i in range(1,N+1) for j in range(1,N+1)])
	val_v = np.log((v_opt + N)/v_opt) - np.sum(tot)
	slope_v = 1/(v_opt+N) - 1/v_opt + np.sum(tot**2)

	print(val_v)
	print(slope_v)
	print(val_v/slope_v)

	v_opt1 = v_opt - val_v/slope_v
	eps = (v_opt1-v_opt)/v_opt
	v_opt = v_opt1

print("Steps: {}".format(step))
d_opt = v_opt*s/N
sigma_est6 = (d_opt+s)/(v_opt+N+3) # same as Q2
print(sigma_est6)
print(v_opt)



