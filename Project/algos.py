import numpy as np 

def P_gaussian(x, mu, sigma, beta):
		return ((1/(2*np.pi*sigma**2)**0.5)*np.exp(-((x-mu)**2)/2/sigma**2))**beta

def likelihood(alpha, x, mu, sigma, beta):
	return ((alpha**beta)*P_gaussian(x, mu[0], sigma[0], beta) + ((1-alpha)**beta)*P_gaussian(x, mu[1], sigma[1], beta))

def ds_error(alpha, mu, sigma, alpha_est, mu_est, sigma_est):
	# Symmetric KL divergence
	if abs(1 - alpha - alpha_est) < abs(alpha - alpha_est):
		mu_est = [mu_est[1], mu_est[0]]
		sigma_est = [sigma_est[1], sigma_est[0]]
	err = 0
	for i in range(len(mu)):
		err += 0.5*((sigma[i]/sigma_est[i])**2 + (sigma_est[i]/sigma[i])**2)
		err += 0.5*((mu_est[i] - mu[i])*(1/sigma[i]**2 + 1/sigma_est[i]**2)*(mu_est[i] - mu[i]))
		err -= 1 # dimension
	return err

class Solver:

	def __init__(self, mu, sigma, alpha):
		self.mu = mu
		self.sigma = sigma
		self.alpha = alpha
	
	def DAEM_GMM(self, X, thresh, mu_est=None, sigma_est=None, alpha_est=None, betas=[0.1,0.6,1.2,1], K=2):
		"""
			Deterministic Annealing EM Algorithm for k=2 Gaussians
		"""
		N = X.size
		errors = []
		alpha_ests = []; mu_ests=[]; likelihoods = []
		beta_step = []
		steps = 0

		# Initial estimates
		if alpha_est is None:
			alpha_est = 1./K
		if mu_est is None:
			mu_est = [X[int(np.random.random()*N)] for j in range(K)]
		if sigma_est is None:
			cov = np.cov(X)
			sigma_est = [cov, cov]

		actual_likelihood = np.sum(np.log(likelihood(self.alpha,X,self.mu,self.sigma, 1))) # With actual parameters

		errors.append(ds_error(self.alpha, self.mu, self.sigma, alpha_est, mu_est, sigma_est)) # error of first estimate

		for beta in betas:	
			print('Maximization for beta = {}'.format(beta))
			llh_01 = likelihood(alpha_est, X, mu_est, sigma_est, 1)
			llh_1 = likelihood(alpha_est, X, mu_est, sigma_est, beta)
			h = (alpha_est**beta)*P_gaussian(X, mu_est[0], sigma_est[0], beta)/llh_1
			tolerance = np.array([1, 1, 1, 1, 1])

			if beta == 1:
				thresh = 1e-10

			while np.any(tolerance >= thresh) and steps <= 100000:
				steps += 1
				# print("Step {}".format(steps))
				llh_00 = llh_01.copy()
				llh_0 = llh_1.copy()

				h_tot = np.sum(h)
				mu_est[0] = np.sum(h*X)/h_tot
				mu_est[1] = np.sum((1-h)*X)/(N-h_tot)

				# Perturb the mu estimates so they split
				if beta!=1:
					mu_est[0] += np.random.normal(mu_est[0], 1e-7)

				sigma_est[0] = (np.sum(h*(X-mu_est[0])**2)/h_tot)**0.5
				sigma_est[1] = (np.sum((1-h)*(X-mu_est[1])**2)/(N - h_tot))**0.5
				alpha_est = h_tot/N

				llh_1 = likelihood(alpha_est, X, mu_est, sigma_est, beta)
				llh_01 = likelihood(alpha_est, X, mu_est, sigma_est, 1)

				h = (alpha_est**beta)*P_gaussian(X, mu_est[0], sigma_est[0], beta)/llh_1
				tolerance = np.abs((llh_01-llh_00)/llh_01)

				errors.append(ds_error(self.alpha, self.mu, self.sigma, alpha_est, mu_est, sigma_est))
				likelihoods.append(np.sum(np.log(llh_01)))
				alpha_ests.append(alpha_est); mu_ests.append(np.array(mu_est))

				# Break based on daem error
				if beta!=1 and np.abs((errors[-1]-errors[-2])/errors[-1])<thresh:
					break
				# elif np.abs((errors[-1]-errors[-2])/errors[-1])<1e-8:
				# 	break	

			beta_step.append((beta, steps-1))
			# print('Beta: {} --- alpha_est: {}, mu_est: {}, sigma_est: {}'.format(beta, alpha_ests, mu_est, sigma_est))

		return alpha_ests, mu_ests, sigma_est, errors, steps, beta_step, likelihoods, actual_likelihood

	def EM_GMM(self, X, thresh, mu_est=None, sigma_est=None, alpha_est=None):
		"""
			Regular Expectation Maximization
		"""
		return self.DAEM_GMM(X=X, thresh=thresh, mu_est=mu_est, sigma_est=sigma_est, alpha_est=alpha_est, betas=[1])
