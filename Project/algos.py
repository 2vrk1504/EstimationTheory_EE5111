import numpy as np 

def P_gaussian(x, mu, sigma, beta):
	n, N = x.shape
	wsig, vsig = np.linalg.eig(sigma)
	sig_inv = vsig.T.dot(np.diag(1/(wsig+1e-9)).dot(vsig)) # sigma inverse
	x_mu = x - mu
	xx = sig_inv[:, 0].reshape((n, 1)) * x_mu[0] 
	for i in range(1,n):
		xx += sig_inv[:, i].reshape((n, 1)) * x_mu[i]
	exp_arg = -np.sum(x_mu * xx, axis=0)/ 2
	val =  ((1/(((2*np.pi)**n)*abs(np.prod(wsig)))**0.5)*np.exp(exp_arg))**beta
	return val

def likelihood(alphas, x, mus, sigmas, beta):
	ll = 0
	K = alphas.size
	for k in range(K):
		ll += (alphas[k]**beta)*P_gaussian(x, mus[k], sigmas[k], beta)
	return ll

def ds_error(n, K, alpha, mu, sigma, alpha_est, mu_est, sigma_est):
	def val(e):
		return e[0]
	# for matching
	alpha = [(alpha[k], k) for k in range(K)]
	alpha_est = [(alpha_est[k], k) for k in range(K)]
	alpha.sort(key=val)
	alpha_est.sort(key=val)

	# Symmetric KL divergence
	err = 0
	for i in range(K):
		k = alpha[i][1]; k1 = alpha_est[i][1]
		sig_inv_k = np.linalg.inv(sigma[k])
		sig_est_inv_k = np.linalg.inv(sigma_est[k1])
		err += 0.5*(np.trace(sig_inv_k.dot(sigma_est[k1]) + sig_est_inv_k.dot(sigma[k])))
		err += 0.5*((mu_est[k1] - mu[k]).T.dot((sig_inv_k + sig_est_inv_k).dot(mu_est[k1] - mu[k])))[0][0]
		err -= n # dimension
	return err

class Solver:

	def __init__(self, mu, sigma, alpha):
		self.mu = mu
		self.sigma = sigma
		self.alpha = alpha
	
	# alternate beta = [0.05,0.1, 0.2,0.5, 0.6, 0.9, 1.2,1.1,1.05,1.0]
	def DAEM_GMM(self, X, thresh, K, mu_est=None, sigma_est=None, alpha_est=None, betas=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.65, 0.75, 0.8, 0.9, 1.0], 
				 history_length=100, tolerance_history_thresh=1e-9, max_steps=10000):
		"""
			Deterministic Anti - Annealing EM Algorithm for k n-dimensional Gaussians
			X.shape = n x N. Xi is n-dimensional. N data points 
		"""
		n, N = X.shape

		errors = []
		alpha_ests = []; mu_ests=[]; likelihoods = []
		beta_step = []
		steps = 0

		# Initial estimates
		if alpha_est is None:
			alpha_est = np.array([1./K for j in range(K)])
		if mu_est is None:
			mu_est = [X[:, int(np.random.random()*N)].reshape((n,1)) for j in range(K)]
		if sigma_est is None:
			sample_mean = np.sum(X, axis=0)/N
			X_mu = X - sample_mean
			cov = np.zeros((n,n))
			for i in range(n):
				cov[i] += np.sum(X_mu[i]*X_mu, axis=1)
			cov /= N
			sigma_est = np.array([np.array(cov) for j in range(K)])

 
		actual_likelihood = np.sum(np.log(likelihood(self.alpha, X, self.mu, self.sigma, 1) + 1e-11))/N # With actual parameters
	
		errors.append(ds_error(n, K, self.alpha, self.mu, self.sigma, alpha_est, mu_est, sigma_est)) # error of first estimate
		likelihoods.append(np.sum(np.log(likelihood(alpha_est, X, mu_est, sigma_est, 1)))/N)
		alpha_ests.append(np.array(alpha_est)); mu_ests.append(np.array(mu_est))

		started = False
		for beta in betas:	
			print('Maximization for beta = {}'.format(beta))

			if beta!=1 and started:
				for k in range(K):
					wsig, vsig = np.linalg.eig(sigma_est[k])
					mu_est[k] = vsig.T.dot(vsig.dot(mu_est[k]) + K*np.random.normal()*wsig.reshape((n,1))**0.5)

			started = True
			llh_01 = likelihood(alpha_est, X, mu_est, sigma_est, 1)
			llh_1 = likelihood(alpha_est, X, mu_est, sigma_est, beta)

			# define h[k, i] = probability that xi belongs to class k
			# however, the following is being done for numerical stability
			h = np.array([(alpha_est[k]**beta)*P_gaussian(X, mu_est[k], sigma_est[k], beta)/(llh_1+1e-9) for k in range(K-1)])
			h = np.append(h, [(1 - np.sum(h, axis=0))], axis=0)
			
			tolerance = np.ones(N)
			tolerance_history = np.ones(history_length)

			if beta == 1:
				thresh = 1e-10
				tolerance_history_thresh = 1e-3

			while tolerance_history[-1] >= thresh and steps <= max_steps:
				steps += 1
				print("Step {}".format(steps), end='\r')
				llh_00 = llh_01.copy()
				llh_0 = llh_1.copy()
				mu_prev = mu_est.copy()

				for k in range(K):
					h_tot_k = np.sum(h[k])
					#print(h[k])
					mu_est[k] = np.sum(h[k]*X, axis=1).reshape((n, 1))/(h_tot_k + 1e-9)

					# Perturb the mu estimates so they split
					# if the max change in the past 100 iterations is not much then
					# if np.max(tolerance_history) <= tolerance_history_thresh:
					# 	wsig, vsig = np.linalg.eig(sigma_est[k]) # can optimize, store once calculated
					# 	mu_est[k] = vsig.dot(vsig.T.dot(mu_est[k]) + 1e-3*np.random.normal()*wsig.reshape((n,1))**0.5) 

					X_mu = X - mu_est[k]
					h_X_mu = h[k]*X_mu
					for i in range(n):
						sigma_est[k][i] = np.sum(X_mu[i]*h_X_mu, axis=1)
					sigma_est[k] /= (h_tot_k + 1e-9)
					sigma_est[k] += 1e-6 * np.eye(n)
					alpha_est[k] = h_tot_k/N

				llh_1 = likelihood(alpha_est, X, mu_est, sigma_est, beta)

				llh_01 = likelihood(alpha_est, X, mu_est, sigma_est, 1)

				# The following is being done for numerical stability
				h = [(alpha_est[k]**beta)*P_gaussian(X, mu_est[k], sigma_est[k], beta)/(llh_1+1e-9) for k in range(K-1)]
				h = np.append(h, [(1 - np.sum(h, axis=0))], axis=0)

				log_ll0 = np.log(llh_00 + 1e-11)
				log_ll1 = np.log(llh_01 + 1e-11)
				tolerance = np.abs(((log_ll0-log_ll1)/log_ll1))
				tolerance_history = np.append(tolerance_history[1:], [np.max(tolerance)])

				errors.append(ds_error(n, K, self.alpha, self.mu, self.sigma, alpha_est, mu_est, sigma_est))
				likelihoods.append(np.sum(log_ll1)/N) 
				alpha_ests.append(np.array(alpha_est)); mu_ests.append(np.array(mu_est))
			print("Steps {} ".format(steps))
			beta_step.append((beta, steps-1))

			# print('Beta: {} --- alpha_est: {}, mu_est: {}, sigma_est: {}'.format(beta, alpha_ests, mu_est, sigma_est))
		return alpha_ests, mu_ests, sigma_est, errors, steps, beta_step, likelihoods, actual_likelihood

	def EM_GMM(self, X, thresh, K, mu_est=None, sigma_est=None, alpha_est=None, max_steps=10000):
		"""
			Regular Expectation Maximization
		"""
		return self.DAEM_GMM(X=X, thresh=thresh, K=K, mu_est=mu_est, sigma_est=sigma_est, alpha_est=alpha_est, betas=[1], max_steps=max_steps)