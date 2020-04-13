import numpy as np 

def P_gaussian(x, mu, sigma, beta):
	n, N = x.shape
	wsig, vsig = np.linalg.eig(sigma)
	sig_inv = vsig.T.dot(np.diag(1/wsig).dot(vsig)) # sigma inverse
	x_mu = x - mu
	xx = sig_inv[:, 0].reshape((n, 1)) * x_mu[0] 
	for i in range(1,n):
		xx += sig_inv[:, i].reshape((n, 1)) * x_mu[i]
	exp_arg = -(x_mu * xx)/ 2
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
	
	def DAEM_GMM(self, X, thresh, mu_est=None, sigma_est=None, alpha_est=None, betas=[0.001, 0.4, 0.9,1.1, 1.0], 
					K=2, history_length=100, tolerance_history_thresh=1e-6):
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
			print(mu_est)
		if sigma_est is None:
			sample_mean = np.sum(X, axis=0)/N
			X_mu = X - sample_mean
			cov = np.zeros((n,n))
			for i in range(n):
				cov[i] += np.sum(X_mu[i]*X_mu, axis=1)
			cov /= N
			sigma_est = [cov for j in range(K)]

 
		actual_likelihood = np.sum(np.log(likelihood(self.alpha, X, self.mu, self.sigma, 1))) # With actual parameters
	
		errors.append(ds_error(n, K, self.alpha, self.mu, self.sigma, alpha_est, mu_est, sigma_est)) # error of first estimate
		likelihoods.append(np.sum(np.log(likelihood(alpha_est, X, mu_est, sigma_est, 1))))
		alpha_ests.append(np.array(alpha_est)); mu_ests.append(np.array(mu_est))

		start = 0
		for beta in betas:	
			print('Maximization for beta = {}'.format(beta))

			if beta!=1 and start!=0:
				noose = sigma_est[0][0,0]**0.5/2
				mu_est[0] += noose #1
				mu_est[1] -= noose #1

			start = 1
			llh_01 = likelihood(alpha_est, X, mu_est, sigma_est, 1)
			llh_1 = likelihood(alpha_est, X, mu_est, sigma_est, beta)

			# define h[k, i] = probability that xi belongs to class k
			h = [(alpha_est[k]**beta)*P_gaussian(X, mu_est[k], sigma_est[k], beta)/llh_1 for k in range(K)]

			# however, the following is being done for numerical stability
			# h = np.array([(alpha_est[k]**beta)*P_gaussian(X, mu_est[k], sigma_est[k], beta)/llh_1 for k in range(K-1)])
			# h = np.append(h, [(1 - np.sum(h, axis=1))], axis=0)
			# NOT NEEDED ^^^^

			tolerance = np.ones(N)
			tolerance_history = np.ones(history_length)

			#for k in range(K):
			#	mu_est[k] += np.random.randn(n, 1)
			
			if beta == 1:
				thresh = 1e-10
				tolerance_history_thresh = 1e-3

			while tolerance_history[-1] >= thresh and steps <= 5000 or steps == 0:
				steps += 1
				#print("Step {}".format(steps), end='\r')
				llh_00 = llh_01.copy()
				llh_0 = llh_1.copy()
				mu_prev = mu_est.copy()

				for k in range(K):
					h_tot_k = np.sum(h[k])
					#print(h[k])
					mu_est[k] = np.sum(h[k]*X, axis=1).reshape((n, 1))/h_tot_k

					# Perturb the mu estimates so they split
					# if the max change in the past 100 iterations is not much then
					if np.max(tolerance_history) <= tolerance_history_thresh:
						mu_est[k] += np.random.randn(n, 1)

					X_mu = X - mu_est[k]
					h_X_mu = h[k]*X_mu
					for i in range(n):
						sigma_est[k][i] = np.sum(X_mu[i]*h_X_mu, axis=1)
					sigma_est[k] /= h_tot_k
					
					alpha_est[k] = h_tot_k/N

				llh_1 = likelihood(alpha_est, X, mu_est, sigma_est, beta)

				llh_01 = likelihood(alpha_est, X, mu_est, sigma_est, 1)

				h = np.array([(alpha_est[k]**beta)*P_gaussian(X, mu_est[k], sigma_est[k], beta)/llh_1 for k in range(K-1)])

				h = np.append(h, [(1 - np.sum(h, axis=1))], axis=0)

				log_ll0 = np.log(llh_00)
				log_ll1 = np.log(llh_01)
				tolerance = np.abs(((log_ll0-log_ll1)/log_ll1))
				tolerance_history = np.append(tolerance_history[1:], [np.max(tolerance)])

				errors.append(ds_error(n, K, self.alpha, self.mu, self.sigma, alpha_est, mu_est, sigma_est))
				likelihoods.append(np.sum(log_ll1))
				alpha_ests.append(np.array(alpha_est)); mu_ests.append(np.array(mu_est))
			print("Steps {} tolerance {} mu {},{}".format(steps,tolerance_history[-1],mu_est[0],mu_est[1]))
			beta_step.append((beta, steps-1))

			# print('Beta: {} --- alpha_est: {}, mu_est: {}, sigma_est: {}'.format(beta, alpha_ests, mu_est, sigma_est))
		return alpha_ests, mu_ests, sigma_est, errors, steps, beta_step, likelihoods, actual_likelihood

	def EM_GMM(self, X, thresh, mu_est=None, sigma_est=None, alpha_est=None):
		"""
			Regular Expectation Maximization
		"""
		return self.DAEM_GMM(X=X, thresh=thresh, mu_est=mu_est, sigma_est=sigma_est, alpha_est=alpha_est, betas=[1])
