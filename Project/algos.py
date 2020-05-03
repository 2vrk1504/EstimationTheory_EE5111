import numpy as np 
import matplotlib.pyplot as plt

def P_gaussian(x, mu, sigma, beta):
	n, N = x.shape
	wsig, vsig = np.linalg.eig(sigma)
	sig_inv = vsig.T.dot(np.diag(1/(wsig)).dot(vsig)) # sigma inverse
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
	
	def mu_sampler(self, X, K):
		n, N = X.shape
		psn = int(K**(1/n)) # perfect a^n = K number... for clustering
		if psn**n < K:
			psn = int(int(psn) + 1)
		range_in_dim = np.empty((n,psn+1))
		for i in range(n):
			range_in_dim[i][0] = np.min(X[i])
			range_in_dim[i][-1] = np.max(X[i])
			c_size = (range_in_dim[i][-1]-range_in_dim[i][0])/psn # approximate cluster size
			for j in range(1,psn):
				range_in_dim[i][j] = range_in_dim[i][j-1] + c_size
		mu_est = []
		def k_in_base_psn(k):
			ii = np.zeros(n, dtype=np.int)
			i = 0
			while k > 0:
				ii[i] = k%psn
				k = k//psn
				i += 1
			return ii
		for k in range(K):
			samples = X
			ii = k_in_base_psn(k)
			for i in range(n):
				samples = samples[:, np.where(np.logical_and(samples[i]<=range_in_dim[i][ii[i]+1],samples[i]>=range_in_dim[i][ii[i]]))[0]]
			if samples.size > 0 and len(mu_est) < K:
				random_indices = np.arange(len(samples))
				np.random.shuffle(random_indices)
				mean = np.sum(samples[:, random_indices[:10]], axis=1).reshape((n,1))/random_indices.size
				mu_est.append(mean)
		if len(mu_est) < K:
			for k in range(len(mu_est), K):
				mu_est.append(X[:, int(np.random.rand()*len(samples))].reshape((n,1)))
		return np.array(mu_est)

	def draw_current(self, mu, sigma, X, K):
		plt.figure('Distribution')
		plt.title('First iteration')
		plt.plot(X[0], X[1], '.', color='blue')
		plt.grid(True)
		theta = np.linspace(0, 2*np.pi, 1000)
		for k in range(K):
			w, v = np.linalg.eig(sigma[k])
			x_pts = 3*(w[0]**0.5)*np.cos(theta)
			y_pts = 3*(w[1]**0.5)*np.sin(theta)
			x1_pts = v[0][0]*x_pts + v[0][1]*y_pts
			y1_pts = v[1][0]*x_pts + v[1][1]*y_pts
			x1_pts += mu[k][0]; y1_pts += mu[k][1];
			plt.plot(x1_pts, y1_pts, color='black', label='DAEM')
		plt.show()

	# alternate beta = [0.05,0.1, 0.2,0.5, 0.6, 0.9, 1.2,1.1,1.05,1.0]
	def DAEM_GMM(self, X, thresh, K, mu_est=None, sigma_est=None, alpha_est=None, betas=[0.8, 0.9, 1.2, 1.0], 
				 history_length=100, min_thresh=1e-10, tolerance_history_thresh=1e-6, max_steps=10000):
		"""
			Deterministic Anti - Annealing EM Algorithm for k n-dimensional Gaussians
			X.shape = n x N. Xi is n-dimensional. N data points 
		"""
		n, N = X.shape

		errors = []
		alpha_ests = []; mu_ests=[]; likelihoods = []
		beta_step = []
		steps = 0
		min_var = 1e-11
		nu = -n-1 + 1e-11 #max(n, 8) # change this according to the prior det(|Sigma_k|)^-p

		# Initial estimates
		if alpha_est is None:
			alpha_est = np.array([1./K for j in range(K)])
		if mu_est is None:
			mu_est = self.mu_sampler(X, K)
		if sigma_est is None:
			sample_mean = np.sum(X, axis=0)/N
			X_mu = X - sample_mean
			cov = np.zeros((n,n))
			for i in range(n):
				cov[i] += np.sum(X_mu[i]*X_mu, axis=1)
			cov /= (N*K*K)
			sigma_est = np.array([np.array(cov) for j in range(K)])
 
		actual_likelihood = np.sum(np.log(likelihood(self.alpha, X, self.mu, self.sigma, 1) + 1e-11))/N # With actual parameters
	
		errors.append(ds_error(n, K, self.alpha, self.mu, self.sigma, alpha_est, mu_est, sigma_est)) # error of first estimate
		likelihoods.append(np.sum(np.log(likelihood(alpha_est, X, mu_est, sigma_est, 1)))/N)
		alpha_ests.append(np.array(alpha_est)); mu_ests.append(np.array(mu_est))

		started = False
		for beta in betas:	
			print('Maximization for beta = {}'.format(beta))

			decomps = [np.linalg.eig(sigma_est[k]) for k in range(K)]

			if beta!=1 and started:
				print('noise added beta change')
				print()
				for k in range(K):
					wsig, vsig = decomps[k]
					mu_est[k] = vsig.T.dot(vsig.dot(mu_est[k]) + np.random.normal(n,1)*wsig.reshape((n,1))**0.5)

			started = True
			llh_01 = likelihood(alpha_est, X, mu_est, sigma_est, 1)
			llh_1 = likelihood(alpha_est, X, mu_est, sigma_est, beta)

			# define h[k, i] = probability that xi belongs to class k
			# however, the following is being done for numerical stability
			h = np.array([(alpha_est[k]**beta)*P_gaussian(X, mu_est[k], sigma_est[k], beta)/(llh_1+1e-9) for k in range(K-1)])
			h = np.append(h, [(1 - np.sum(h, axis=0))], axis=0)
			
			tolerance = np.ones(N)
			tolerance_history = np.ones(history_length)
			ll_history = np.ones(history_length)

			if beta == 1:
				thresh = min_thresh
				tolerance_history_thresh = min_thresh*10

			while tolerance_history[-1] >= thresh and steps <= max_steps:
				steps += 1
				print("Step {}".format(steps), end='\r')
				llh_00 = llh_01.copy()
				llh_0 = llh_1.copy()

				for k in range(K):
					h_tot_k = np.sum(h[k])
					#print(h[k])
					mu_est[k] = np.sum(h[k]*X, axis=1).reshape((n, 1))/(h_tot_k + 1e-19)

					X_mu = X - mu_est[k]
					h_X_mu = h[k]*X_mu
					for i in range(n):
						sigma_est[k][i] = np.sum(X_mu[i]*h_X_mu, axis=1)
					sigma_est[k] += min_var * np.eye(n)
					sigma_est[k] /= (h_tot_k + nu + n + 1)
					# if np.min(np.diag(sigma_est[k])) < min_var:
					# 	sigma_est[k] += min_var * np.eye(n)
					# sigma_est[k] = 6 * np.eye(n)
					alpha_est[k] = h_tot_k/N

				decomps = [np.linalg.eig(sigma_est[k]) for k in range(K)]

				# Perturb the mu estimates so they split
				# if the max change in the past 100 iterations is not much then
				if np.max(tolerance_history) <= tolerance_history_thresh:
					wsig, vsig = decomps[k] # can optimize, store once calculated
					print('noise added because of history')
					mu_est[k] = vsig.dot(vsig.T.dot(mu_est[k]) + np.random.normal(n,1)*wsig.reshape((n,1))**0.5) 


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
				ll_history = np.append(ll_history[1:], [likelihoods[-1]-likelihoods[-2]])
				alpha_ests.append(np.array(alpha_est)); mu_ests.append(np.array(mu_est))

				# if oscillations take place
				if np.where(ll_history < 0)[0].size >= 33 and ll_history[-1]>0:
					if beta != 1:
						print('ll break')
						break
					else:
						for k in range(K):
							wsig, vsig = decomps[k]
							# print('noise added because of ll_history')
							mu_est[k] = vsig.dot(vsig.T.dot(mu_est[k]) + np.random.normal(n,1)*wsig.reshape((n,1))**0.5)


			print("Steps {} ".format(steps))
			beta_step.append((beta, steps-1))

			# print('Beta: {} --- alpha_est: {}, mu_est: {}, sigma_est: {}'.format(beta, alpha_ests, mu_est, sigma_est))
		return alpha_ests, mu_ests, sigma_est, errors, steps, beta_step, likelihoods, actual_likelihood

	def EM_GMM(self, X, thresh, K, mu_est=None, sigma_est=None, alpha_est=None, max_steps=10000):
		"""
			Regular Expectation Maximization
		"""
		return self.DAEM_GMM(X=X, thresh=thresh, min_thresh=thresh, K=K, mu_est=mu_est, sigma_est=sigma_est, alpha_est=alpha_est, betas=[1], max_steps=max_steps)



	def fit_data(self, X, thresh=1e-10, min_thresh=1e-10, max_steps=5000, Kmin=2, Kmax=10, algo='DAEM'):
		"""
			Bayesian Information Criterion to best fit data
		"""
		n, N = X.shape
		min_BIC = 1e20 # start with max
		k_best = Kmin
		if algo == 'EM':
			for k in range(Kmin, Kmax+1):
				result = self.EM_GMM(X=X, thresh=thresh, K=k, max_steps=5000)
				bic = 2*k*np.log(N) - 2*result[6][-1]*N
				if bic < min_BIC:
					k_best = k
					min_BIC = bic

		else:
			for k in range(Kmin, Kmax+1):
				result = self.DAEM_GMM(X=X, thresh=thresh, min_thresh=min_thresh, K=k, max_steps=5000)
				bic = 2*k*np.log(N) - 2*result[6][-1]*N
				if bic < min_BIC:
					k_best = k
					min_BIC = bic

		return k_best, result


