import numpy as np 
import matplotlib.pyplot as plt 
from algos import Solver

N = 1000		# number of samples
K = 2			# number of mixed Gaussians
mus = [0]		
sigma = np.array([		# covariance matrices
	np.array([[6.25]]), 
	np.array([[6.25]])
])	

def toss(alpha):
	x = np.random.random()
	last = 0
	for k in range(len(alpha)):
		if x >= last and x<last+alpha[k]:
			return k
		else:
			last += alpha[k]

alphas = [[0.5, 0.5]] #np.linspace(0.6, 0.6, 1) # mixing coefficients

colors = ['red', 'pink', 'brown']


for mu_iter in mus:
	mu = np.array([[55],[65]]) #np.array([[-mu_iter], [mu_iter]])
	errorss_daem = []; alphass_daem = []; muss_daem = [];
	errorss_em = []; alphass_em = []; muss_em = []
	bs = []

	for alpha in alphas:
		solver = Solver(alpha=np.array(alpha), mu=mu, sigma=sigma)

		# Xi is 1 dimensional. N data points
		# need to generate data properly as required 
		X = np.array([np.random.normal(mu[toss(alpha)], sigma[toss(alpha)]**0.5) for j in range(N)]).reshape(1, N)

		print('Actual:')
		print('alpha: {}\nmu:\n{}\nsigma:\n{}\n\n'.format(alpha, mu, sigma))
		alpha_est_daem, mu_est_daem, sigma_est_daem, errors_daem, steps, beta_step, likelihoods_daem, actual_likelihood_daem = solver.DAEM_GMM(X=X, thresh=1e-6, K=K)
		errorss_daem.append(errors_daem)
		alphass_daem.append(alpha_est_daem)
		muss_daem.append(mu_est_daem)
		bs.append(beta_step)
		print('DAEM')
		print('Steps:\n{}\n alpha_est: {}\nmu_est:\n{}\nsigma_est:\n{}\n\n'.format(steps, alpha_est_daem[-1], mu_est_daem[-1], sigma_est_daem))
		alpha_est_em, mu_est_em, sigma_est_em, errors_em, steps, __, likelihoods_em, actual_likelihood_em = solver.EM_GMM(X=X, thresh=1e-10, K=K)
		errorss_em.append(errors_em)
		alphass_em.append(alpha_est_em)
		muss_em.append(mu_est_em)
		print('EM')
		print('Steps:\n{}\nalpha_est: {}\nmu_est:\n{}\nsigma_est:\n{}\n\n'.format(steps, alpha_est_em[-1], mu_est_em[-1], sigma_est_em))
		print()

	################## DAEM PLOTS ###############################

	plt.figure('Error')
	plt.subplot(1,2,1)
	plt.title(r'DAEM Error vs. Iterations, $(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
	for i, alpha in enumerate(alphas):
		plt.plot(errorss_daem[i][1:], 'x-', label=r'$\alpha=$'+str(alpha))
		for _bs in bs[i]:
			plt.axvline(x=_bs[1], color=colors[i], ls=':', lw=1, label=r'$\beta=$'+str(_bs[0]))
	plt.grid(True)
	plt.legend(loc='upper right')

	plt.figure('alpha')
	plt.subplot(1,2,1)
	plt.title(r'DAEM, $\hat{\alpha}$ vs. Iterations, $(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
	alphass_daem = np.array(alphass_daem)
	for i, alpha in enumerate(alphas):
		plt.plot(alphass_daem[i], label=r'$\alpha=$'+str(alpha))
		for _bs in bs[i]:
			plt.axvline(x=_bs[1], color=colors[i], ls=':', lw=1, label=r'$\beta=$'+str(_bs[0]))
	plt.grid(True)
	plt.legend(loc='upper right')
	'''
	plt.figure('Mu')
	plt.subplot(1,2,1)
	muss_daem = np.array(muss_daem)
	plt.title(r'DAEM, $\hat{\mu}$ vs. Iterations, $(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
	for i, alpha in enumerate(alphas):
		plt.plot(muss_daem[i][:,0][:,0,0], muss_daem[i][:,1][:,0,0], 'yx-', label=r'$\alpha=$'+str(alpha))
		for _bs in bs[i]:
			plt.plot(muss_daem[i][_bs[1],0][0,0], muss_daem[i][_bs[1],1][0,0], 'rx') #, label=r'$\beta=$'+str(_bs[0]))
		plt.plot(muss_daem[i][-1][0][0,0], muss_daem[i][-1][1][0,0],'x-', color='green')
	plt.grid(True)
	plt.legend(loc='upper right')
	'''
	plt.figure('Likelihood')
	plt.subplot(1,2,1)
	plt.title(r'DAEM Likelihoods vs. Iterations, $(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
	for i, alpha in enumerate(alphas):
		plt.plot(likelihoods_daem, label=r'$\alpha=$'+str(alpha))
	plt.plot(np.repeat(actual_likelihood_daem,len(likelihoods_daem)), label='Actual')	
	plt.grid(True)
	plt.legend(loc='upper right')

	##################### EM PLOTS #################################

	plt.figure('Error')
	plt.subplot(1,2,2)
	plt.title(r'EM Error vs. Iterations, $(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
	for i, alpha in enumerate(alphas):
		plt.plot(errorss_em[i], 'x-', label=r'$\alpha=$'+str(alpha))
	plt.grid(True)
	plt.legend(loc='upper right')

	plt.figure('alpha')
	plt.subplot(1,2,2)
	plt.title(r'EM, $\hat{\alpha}$ vs. Iterations, $(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
	alphass_em = np.array(alphass_em)
	for i, alpha in enumerate(alphas):
		plt.plot(alphass_em[i], label=r'$\alpha=$'+str(alpha))
	plt.grid(True)
	plt.legend(loc='upper right')
	
	'''
	plt.figure('Mu')
	plt.subplot(1,2,2)
	muss_em = np.array(muss_em)
	plt.title(r'EM, $\hat{\mu}$ vs. Iterations, $(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
	for i, alpha in enumerate(alphas):
		plt.plot(muss_em[i][:, 0][:,0,0], muss_em[i][:, 1][:,0,0], 'yx-', label=r'$\alpha=$'+str(alpha))
		plt.plot(muss_em[i][-1][0][0,0], muss_em[i][-1][1][0,0],'x-', color='green')
	plt.grid(True)
	plt.legend(loc='upper right')
	'''
	plt.figure('Likelihood')
	plt.subplot(1,2,2)
	plt.title(r'EM Likelihoods vs. Iterations, $(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
	for i, alpha in enumerate(alphas):
		plt.plot(likelihoods_em, label=r'$\alpha=$'+str(alpha))
	plt.plot(np.repeat(actual_likelihood_em,len(likelihoods_em)), label='Actual')	
	plt.grid(True)
	plt.legend(loc='upper right')

	################# END OF PLOTS ################

	plt.show()

		# stepss.append(steps);
	# stepsss.append(stepss)
	# print(stepss)

# plt.figure()
# plt.title(r'iterations vs. $\alpha$')
# for j, stepss in enumerate(stepsss):
# 	plt.semilogy(alphas, stepss, '^-', label=r'$\mu$='+str(mus[j]))
# 	# plt.semilogy(alphas, stepss)
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.show()