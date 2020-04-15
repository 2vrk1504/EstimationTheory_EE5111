import numpy as np 
import matplotlib.pyplot as plt 
from algos import Solver
# from algos_arvind import Solver


N = 2000		# number of samples
K = 2			# number of mixed Gaussians
n = 2			# dimension
mu = np.array([
	np.array([[-20],
			  [0]]),
	np.array([[10],
			  [0]]),
	# np.array([[0],
	# 		  [5]]),
	# np.array([[-5],
	# 		  [0]])
	])	
sigma = np.array([		# covariance matrices
	np.array([[100, 0],
			  [0, 100]]), 
	np.array([[100, 0],
			  [0, 100]]), 
	# np.array([[1, 0.95],
	# 		  [0.95, 1]]), 
	# np.array([[1, 0.95],
	# 		  [0.95, 1]])
])	

alphas = [[0.1, 0.9]] #np.linspace(0.6, 0.6, 1) # mixing coefficients

def toss(alpha):
	x = np.random.random()
	last = 0
	for k in range(len(alpha)):
		if x >= last and x<last+alpha[k]:
			return k
		else:
			last += alpha[k]

colors = ['red', 'pink', 'brown']
color_dis = ['green', 'blue']
 
errorss_daem = []; alphass_daem = []; muss_daem = [];
errorss_em = []; alphass_em = []; muss_em = []
bs = []

for alpha in alphas:
	solver = Solver(alpha=np.array(alpha), mu=mu, sigma=sigma)

	# Xi is 1 dimensional. N data points
	# need to generate data properly as required 
	X = np.empty((n,0))
	X_classes = [np.empty((n,0)) for k in range(K)]
	decomps = [np.linalg.eig(sigma[k]) for k in range(K)]
	for j in range(N):
		k_chosen = toss(alpha)
		my_mu = mu[k_chosen]
		wsig, vsig = decomps[k_chosen]
		sample = my_mu + vsig.dot((wsig**0.5)*np.random.randn(n, 1))
		X_classes[k_chosen] = np.append(X_classes[k_chosen], sample, axis=1)
		X = np.append(X, sample, axis=1)


	def coco(X):
		sample_mean = np.sum(X, axis=0)/N
		X_mu = X - sample_mean
		cov = np.zeros((n,n))
		for i in range(n):
			cov[i] += np.sum(X_mu[i]*X_mu, axis=1)
		cov /= N
		return cov

	print(coco(X_classes[0]))
	print(coco(X_classes[1]))

	print('Actual:')
	print('alpha: {}\nmu:\n{}\nsigma:\n{}\n\n'.format(alpha, mu, sigma))
	alpha_est_daem, mu_est_daem, sigma_est_daem, errors_daem, steps, beta_step, likelihoods_daem, actual_likelihood_daem = solver.DAEM_GMM(X=X, thresh=1e-4, K=K, max_steps=5000)
	errorss_daem.append(errors_daem)
	alphass_daem.append(alpha_est_daem)
	muss_daem.append(mu_est_daem)
	bs.append(beta_step)
	print('DAEM')
	print('Steps:\n{}\n alpha_est: {}\nmu_est:\n{}\nsigma_est:\n{}\n\n'.format(steps, alpha_est_daem[-1], mu_est_daem[-1], sigma_est_daem))
	alpha_est_em, mu_est_em, sigma_est_em, errors_em, steps, __, likelihoods_em, actual_likelihood_em = solver.EM_GMM(X=X, thresh=1e-10, K=K, max_steps=5000)
	errorss_em.append(errors_em)
	alphass_em.append(alpha_est_em)
	muss_em.append(mu_est_em)
	print('EM')
	print('Steps:\n{}\nalpha_est: {}\nmu_est:\n{}\nsigma_est:\n{}\n\n'.format(steps, alpha_est_em[-1], mu_est_em[-1], sigma_est_em))
	print()

################## DAEM PLOTS ###############################

plt.figure('Error')
plt.subplot(1,2,1)
plt.title(r'DAEM Error vs. Iterations,') #$(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
for i, alpha in enumerate(alphas):
	plt.plot(errorss_daem[i][1:], 'x-', label=r'$\alpha=$'+str(alpha))
	for _bs in bs[i]:
		plt.axvline(x=_bs[1], color=colors[i], ls=':', lw=1, label=r'$\beta=$'+str(_bs[0]))
plt.grid(True)
plt.legend(loc='upper right')

plt.figure('alpha')
plt.subplot(1,2,1)
plt.title(r'DAEM, $\hat{\alpha}$ vs. Iterations, ')#$(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
alphass_daem = np.array(alphass_daem)
for i, alpha in enumerate(alphas):
	for k in range(len(alpha)):
		plt.plot(alphass_daem[i][:,k], label=r'$\alpha=$'+str(alpha[k]))
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
plt.title(r'DAEM Likelihoods vs. Iterations, ')#$(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
for i, alpha in enumerate(alphas):
	plt.plot(likelihoods_daem, label=r'$\alpha=$'+str(alpha))
plt.plot(np.repeat(actual_likelihood_daem,len(likelihoods_daem)), label='Actual')	
plt.grid(True)
plt.legend(loc='upper right')

##################### EM PLOTS #################################

plt.figure('Error')
plt.subplot(1,2,2)
plt.title(r'EM Error vs. Iterations, ')#$(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
for i, alpha in enumerate(alphas):
	plt.plot(errorss_em[i], 'x-', label=r'$\alpha=$'+str(alpha))
plt.grid(True)
plt.legend(loc='upper right')

plt.figure('alpha')
plt.subplot(1,2,2)
plt.title(r'EM, $\hat{\alpha}$ vs. Iterations, ')#$(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
alphass_em = np.array(alphass_em)
for i, alpha in enumerate(alphas):
	for k in range(len(alpha)):
		plt.plot(alphass_em[i][:,k], label=r'$\alpha=$'+str(alpha[k]))
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
plt.title(r'EM Likelihoods vs. Iterations, ')#$(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
for i, alpha in enumerate(alphas):
	plt.plot(likelihoods_em, label=r'$\alpha=$'+str(alpha))
plt.plot(np.repeat(actual_likelihood_em,len(likelihoods_em)), label='Actual')	
plt.grid(True)
plt.legend(loc='upper right')


############## DISTRIBUTION ####################

plt.figure('Distribution')
plt.title(r'Distribution and prediction')#$(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
for i, alpha in enumerate(alphas):
	#for j in range(N):
	for k in range(K):
		plt.plot(X_classes[k][0], X_classes[k][1], '.', color=color_dis[k])
	
	theta = np.linspace(0, 2*np.pi, 1000)
	for k in range(K):
		w, v = np.linalg.eig(sigma_est_daem[k])
		x_pts = 3*(w[0]**0.5)*np.cos(theta)
		y_pts = 3*(w[1]**0.5)*np.sin(theta)
		x1_pts = v[0][0]*x_pts + v[0][1]*y_pts
		y1_pts = v[1][0]*x_pts + v[1][1]*y_pts
		x1_pts += mu_est_daem[-1][k][0]; y1_pts += mu_est_daem[-1][k][1];
		plt.plot(x1_pts, y1_pts, color='black', label='DAEM')
		w1, v1 = np.linalg.eig(sigma_est_em[k])
		x_pts1 = 3*(w1[0]**0.5)*np.cos(theta)
		y_pts1 = 3*(w1[1]**0.5)*np.sin(theta)
		x1_pts1 = v1[0][0]*x_pts1 + v[0][1]*y_pts1
		y1_pts1 = v1[1][0]*x_pts1 + v[1][1]*y_pts1
		x1_pts1 += mu_est_em[-1][k][0]; y1_pts1 += mu_est_em[-1][k][1];
		plt.plot(x1_pts1, y1_pts1, color='red', label='EM')
plt.ylim([-100, 100])
plt.xlim([-100, 100])
plt.legend(loc='upper right')
plt.grid(True)


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