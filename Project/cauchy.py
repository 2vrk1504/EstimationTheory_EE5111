import numpy as np
import matplotlib.pyplot as plt



import numpy as np 
import matplotlib.pyplot as plt 
from algos_arvind import Solver

print('START')

N = 5000 		# number of samples
K = 10			# number of mixed Gaussians
n = 1			# dimension

g=5

X = g*(np.random.standard_cauchy(N)).reshape((1,N))
#s = s[(s>-25) & (s<25)]  # truncate distribution so it plots well


mu=np.zeros(K).reshape((1,K))
sigma=np.ones(K).reshape((1,K))
colors = ['red', 'pink', 'brown','gold']

errorss_daem = []; alphass_daem = []; muss_daem = [];
errorss_em = []; alphass_em = []; muss_em = []
bs = []

alpha=[1/K for i in range(K)]
alphas=[alpha]
solver = Solver(alpha=np.array(alpha), mu=mu, sigma=sigma)


#Data generation



print('Actual:')
print('alpha: {}\nmu:\n{}\nsigma:\n{}\n\n'.format(alpha, mu, sigma))
#alpha_est_daem, mu_est_daem, sigma_est_daem, errors_daem, steps, beta_step, likelihoods_daem, actual_likelihood_daem = solver.DAEM_GMM(X=X, thresh=1e-6, K=K, max_steps=5000)
alpha_est_daem, mu_est_daem, sigma_est_daem, steps, beta_step, likelihoods_daem = solver.DAEM_GMM(X=X, thresh=1e-6, K=K, max_steps=10000)
#errorss_daem.append(errors_daem)
alphass_daem.append(alpha_est_daem)
muss_daem.append(mu_est_daem)
bs.append(beta_step)
print('DAEM')
print('Steps:\n{}\n alpha_est: {}\nmu_est:\n{}\nsigma_est:\n{}\n\n'.format(steps, alpha_est_daem[-1], mu_est_daem[-1], sigma_est_daem))

#alpha_est_em, mu_est_em, sigma_est_em, errors_em, steps, __, likelihoods_em, actual_likelihood_em = solver.EM_GMM(X=X, thresh=1e-10, K=K, max_steps=5000)
alpha_est_em, mu_est_em, sigma_est_em, steps, __, likelihoods_em = solver.EM_GMM(X=X, thresh=1e-10, K=K, max_steps=10000)
#errorss_em.append(errors_em)
alphass_em.append(alpha_est_em)
muss_em.append(mu_est_em)
print('EM')
print('Steps:\n{}\nalpha_est: {}\nmu_est:\n{}\nsigma_est:\n{}\n\n'.format(steps, alpha_est_em[-1], mu_est_em[-1], sigma_est_em))
print()

'''

plt.figure('Likelihood')
plt.subplot(1,2,1)
plt.title(r'DAEM Likelihoods vs. Iterations, ')#$(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
for i, alpha in enumerate(alphas):
	plt.plot(likelihoods_daem, label=r'$\alpha=$'+str(alpha))
#plt.plot(np.repeat(actual_likelihood_daem,len(likelihoods_daem)), label='Actual')	
plt.grid(True)
#plt.legend(loc='upper right')

plt.figure('Likelihood')
plt.subplot(1,2,2)
plt.title(r'EM Likelihoods vs. Iterations, ')#$(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
for i, alpha in enumerate(alphas):
	plt.plot(likelihoods_em, label=r'$\alpha=$'+str(alpha))
#plt.plot(np.repeat(actual_likelihood_em,len(likelihoods_em)), label='Actual')	
plt.grid(True)
#plt.legend(loc='upper right')

'''

#plotting both estimates


s=X

s = s[(s>-25) & (s<25)]


print("mu est",mu_est_daem[-1])


plt.figure("DAEM")

mu_est=np.array(mu_est_daem[-1])


mm=[]
ss=[]
for i in range(K):
	mm.append(mu_est[i][0][0])
	ss.append(sigma_est_daem[i][0][0])


a=alpha_est_daem[-1]


xx=np.linspace(-50,50,N)
yy=np.zeros(N)


plt.plot(xx,1/(np.pi*g*(1+(xx/g)**2)),label="Cauchy (Gamma = "+str(g)+")")

for i in range(K):
    
    yy+=(a[i]/(2*np.pi*ss[i])**0.5)*np.exp(-(xx - mm[i])**2/ss[i])

plt.plot(xx,yy,label=" DAEM Estimate (K ="+str(K)+")")
plt.grid(True)
plt.legend(loc=2)


plt.figure("EM")
mu_est=np.array(mu_est_em[-1])


mm=[]
ss=[]
for i in range(K):
	mm.append(mu_est[i][0][0])
	ss.append(sigma_est_em[i][0][0])


a=alpha_est_em[-1]

yy=np.zeros(N)


plt.plot(xx,1/(np.pi*g*(1+(xx/g)**2)),label="Cauchy (Gamma = "+str(g)+")")

for i in range(K):
    
    yy+=(a[i]/(2*np.pi*ss[i])**0.5)*np.exp(-(xx - mm[i])**2/ss[i])

plt.plot(xx,yy,label="EM Estimate (K ="+str(K)+")")
plt.grid(True)
plt.legend(loc=2)
plt.show()

