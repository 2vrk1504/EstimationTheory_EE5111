import numpy as np 
import matplotlib.pyplot as plt 
from algos import Solver

print('START')

N = 1000 		# number of samples
K = 10			# number of mixed Gaussians
n = 1			# dimension

g=10

mu=np.zeros((K,1,1))
sigma=np.ones((K,1,1))
colors = ['red', 'pink', 'brown','gold']

errorss_daem = []; alphass_daem = []; muss_daem = [];
errorss_em = []; alphass_em = []; muss_em = []
bs = []

alpha=[1/K for i in range(K)]
alphas=[alpha]
solver = Solver(alpha=np.array(alpha), mu=mu, sigma=sigma)


#Data generation
X = g*(np.random.standard_cauchy(N)).reshape((n,N))


# print('Actual:')
# print('alpha: {}\nmu:\n{}\nsigma:\n{}\n\n'.format(alpha, mu, sigma))
#alpha_est_daem, mu_est_daem, sigma_est_daem, errors_daem, steps, beta_step, likelihoods_daem, actual_likelihood_daem = solver.DAEM_GMM(X=X, thresh=1e-6, K=K, max_steps=5000)
alpha_est_daem, mu_est_daem, sigma_est_daem, er, steps, beta_step, likelihoods_daem, al = solver.DAEM_GMM(X=X, thresh=1e-6, K=K, max_steps=10000)
#errorss_daem.append(errors_daem)
alphass_daem.append(alpha_est_daem)
muss_daem.append(mu_est_daem)
bs.append(beta_step)
print('DAEM')
print('Steps:\n{}\n alpha_est: {}\nmu_est:\n{}\nsigma_est:\n{}\n\n'.format(steps, alpha_est_daem[-1], mu_est_daem[-1], sigma_est_daem))

#alpha_est_em, mu_est_em, sigma_est_em, errors_em, steps, __, likelihoods_em, actual_likelihood_em = solver.EM_GMM(X=X, thresh=1e-10, K=K, max_steps=5000)
alpha_est_em, mu_est_em, sigma_est_em, er, steps, __, likelihoods_em, al = solver.EM_GMM(X=X, thresh=1e-10, K=K, max_steps=10000)
#errorss_em.append(errors_em)
alphass_em.append(alpha_est_em)
muss_em.append(mu_est_em)
print('EM')
print('Steps:\n{}\nalpha_est: {}\nmu_est:\n{}\nsigma_est:\n{}\n\n'.format(steps, alpha_est_em[-1], mu_est_em[-1], sigma_est_em))
print()


actual_likelihood = np.sum(np.log(1/(np.pi*g*(1+(X/g)**2))))/N

plt.figure('Likelihood')
plt.subplot(1,2,1)
plt.title(r'DAEM Likelihoods vs. Iterations, ')#$(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
for i, alpha in enumerate(alphas):
	plt.plot(likelihoods_daem)
plt.plot(np.repeat(actual_likelihood,len(likelihoods_daem)), label='Actual')	
plt.grid(True)
#plt.legend(loc='upper right')

plt.figure('Likelihood')
plt.subplot(1,2,2)
plt.title(r'EM Likelihoods vs. Iterations, ')#$(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
for i, alpha in enumerate(alphas):
	plt.plot(likelihoods_em)
plt.plot(np.repeat(actual_likelihood,len(likelihoods_em)), label='Actual')	
plt.grid(True)
# plt.legend(loc='upper right')



#plotting both estimates
s=X
xx=np.linspace(-1000,1000,100*N)
dx=2000/100/N
yy_daem=np.zeros(100*N)
yy_em=np.zeros(100*N)

cauchy_pdf = 1/(np.pi*g*(1+(xx/g)**2))

for i in range(K):
	yy_daem += ((alpha_est_daem[-1][i]/(2*np.pi*sigma_est_daem[i])**0.5)*np.exp(-(xx - mu_est_daem[-1][i])**2/sigma_est_daem[i]))[0]
	yy_em += ((alpha_est_em[-1][i]/(2*np.pi*sigma_est_em[i])**0.5)*np.exp(-(xx - mu_est_em[-1][i])**2/sigma_est_em[i]))[0]


# KL Divergence
dkl_daem = np.sum(cauchy_pdf*dx*np.log(cauchy_pdf/yy_daem))
dkl_em = np.sum(cauchy_pdf*dx*np.log(cauchy_pdf/yy_em))

print()
print('KL Divergence')
print('DAEM KL Divergence: {}'.format(dkl_daem))
print('EM KL Divergence: {}'.format(dkl_em))

plt.figure("DAEM")
plt.plot(xx, cauchy_pdf,label="Cauchy (Gamma = "+str(g)+")")
plt.plot(xx,yy_daem,label=" DAEM Estimate (K ="+str(K)+")")
plt.grid(True)
plt.legend(loc=2)

plt.figure("EM")
plt.plot(xx,cauchy_pdf,label="Cauchy (Gamma = "+str(g)+")")
plt.plot(xx,yy_em,label="EM Estimate (K ="+str(K)+")")
plt.grid(True)
plt.legend(loc=2)
plt.show()
