import numpy as np 
import matplotlib.pyplot as plt

# Noise Params
NOISE_PDF_NAME = "2-Pareto"
g = 0.1 # gamma


# define your pdf here
def noise_pdf(x):
	return np.where(x<=1, 0, 1/(x**2))

#inverse cdf for sampling
def cdf_inv(x): 
	return g/(1-x)


# TRUE MODEL PARAMETERS
QPSK_SYMBOLS = [1+1j, -1+1j, 1-1j, -1-1j]
N = 128		# dimensions of signal detected
L = 32 		# channel taps

F = np.array([ [ np.exp((np.pi*2j*i*j)/N) for j in range(L)] for i in range(N)])
FH = F.conjugate().T

c_sigma = 0.5 ** 0.5 	# channel parameter
lambda_ = 0.2			# channel parameter

p = np.exp(-lambda_ * (np.arange(1,L+1) - 1)).reshape((L,1))
a = np.random.normal(0, c_sigma, (L,1))
b = np.random.normal(0, c_sigma, (L,1))
norm_p = np.sum(p**2)

# s0 is the true channel impulse response vector
s0 = ((a + 1j*b)*p)/norm_p	# h[k] = (a[k] + jb[k])p[k] / norm(p)

def plot_channel(hplot, label):
	plt.figure()
	plt.subplot(2,1,1)
	plt.title(label + ' Real and Imaginary')
	plt.grid(True)
	plt.stem([(hh[0]).real for hh in s0], linefmt='g-', markerfmt='go', label='Original', use_line_collection=True)
	plt.stem([(hh[0]).real for hh in hplot], linefmt='r:', markerfmt='ro', label=label, use_line_collection=True)
	plt.legend(loc='upper right')
	plt.subplot(2,1,2)
	plt.grid(True)
	plt.stem([(hh[0]).imag for hh in s0], linefmt='g-', markerfmt='go', label='Original', use_line_collection=True)
	plt.stem([(hh[0]).imag for hh in hplot], linefmt='r:', markerfmt='ro', label=label, use_line_collection=True)
	plt.legend(loc='upper right')

def plot_MSE(hplot, label):
	plt.figure('MSE')
	plt.title('MSE vs. iters')
	plt.grid(True)
	MSE = np.sum(np.abs(hplot-s0)**2, axis=1)
	print('MSE of ' + label + '= ', MSE[-1])
	plt.plot(MSE, label=label)
	plt.legend(loc='upper right')

X = np.array([QPSK_SYMBOLS[int(np.random.rand()*4)] for i in range(N)])
A = np.diag(X).dot(F)
AH = np.conjugate(A).T

noise = cdf_inv(np.random.rand(N)).reshape((N,1)) + 1j*cdf_inv(np.random.rand(N)).reshape((N,1))

# received symbols
y = A.dot(s0) + noise

# LS Estimate
s_ls = FH.dot((y.flatten()/X).reshape((N,1)))/N


# Assuming GMM Noise
K = 4 # number of clusters


def P_gaussian(x, mu, sigma, beta):
	N = x.size
	x_mu = np.abs(x - mu)
	exp_arg = 0.5*(x_mu**2)/sigma
	val =  (np.exp(-exp_arg)/(2*np.pi*sigma)**0.5)**beta
	return val

def likelihood(alphas, x, mus, sigmas, beta):
	ll = 0
	K = alphas.size
	for k in range(K):
		ll += (alphas[k]**beta)*P_gaussian(x, mus[k], sigmas[k], beta)
	return ll

def solve(y, thresh, K, s_est, mu_est=None, sigma_est=None, alpha_est=None, betas=[0.5, 0.8, 1.2, 1.0], history_length=100, min_thresh=1e-10, tolerance_history_thresh=1e-6, max_steps=10000):
	N = y.size
	likelihoods = []
	beta_step = []
	steps = 0
	s_ests = [s_est.copy()]
	ww = (y-A.dot(s_est)).reshape(N)

	# Initial estimates
	if alpha_est is None:
		alpha_est = np.array([1./K for j in range(K)])

	if mu_est is None:
		mu_est = np.array([ww[int(np.random.rand()*N)] for j in range(K)])

	if sigma_est is None:
		sample_mean = np.sum(y-A.dot(s_est), axis=0)/N
		y_mu = np.abs(y-A.dot(s_est) - sample_mean)
		cov = np.sum(y_mu*y_mu)
		cov /= (N*K*K)
		sigma_est = np.array([cov for j in range(K)])

	likelihoods.append(np.sum(np.log(likelihood(alpha_est, ww, mu_est, sigma_est, 1)))/N)

	started = False
	for beta in betas:	
		print('Maximization for beta = {}'.format(beta))

		if beta!=1 and started:
			print('noise added beta change')
			print()
			for k in range(K):
				mu_est[k] += np.random.normal()*sigma_est[k]**0.5

		started = True
		llh_01 = likelihood(alpha_est, ww, mu_est, sigma_est, 1)
		llh_1 = likelihood(alpha_est, ww, mu_est, sigma_est, beta)

		# define h[k, i] = probability that xi belongs to class k
		# however, the following is being done for numerical stability
		h = np.array([(alpha_est[k]**beta)*P_gaussian(ww, mu_est[k], sigma_est[k], beta)/(llh_1+1e-9) for k in range(K-1)])
		h = np.append(h, [(1 - np.sum(h, axis=0))], axis=0)
		h_tot = np.array([np.sum(h[k]) for k in range(K)]) + 1e-20
		
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

			B = np.zeros((N,N))
			for k in range(K):
				B += (np.diag(h[k])-h[k].reshape((N,1)).dot(h[k].reshape((1,N)))/h_tot[k])/sigma_est[k]

			s_est = np.linalg.inv(AH.dot(B).dot(A)).dot(AH).dot(B).dot(y)
			ww = (y-A.dot(s_est)).reshape(N)

			for k in range(K):
				mu_est[k] = np.sum(h[k]*ww)/(h_tot[k] + 1e-19)
				sigma_est[k] = np.sum((np.abs(ww-mu_est[k])**2)*h[k]) + 1e-20
				sigma_est[k] /= (h_tot[k])
				alpha_est[k] = h_tot[k]/N


			# Perturb the mu estimates so they split
			# if the max change in the past 100 iterations is not much then
			if np.max(tolerance_history) <= tolerance_history_thresh:
				print('noise added because of history')
				mu_est[k] += np.random.normal()*sigma_est[k]**0.5


			llh_1 = likelihood(alpha_est, ww, mu_est, sigma_est, beta)
			llh_01 = likelihood(alpha_est, ww, mu_est, sigma_est, 1)

			# The following is being done for numerical stability
			h = [(alpha_est[k]**beta)*P_gaussian(ww, mu_est[k], sigma_est[k], beta)/(llh_1+1e-9) for k in range(K-1)]
			h = np.append(h, [(1 - np.sum(h, axis=0))], axis=0)
			h_tot = np.array([np.sum(h[k]) for k in range(K)]) + 1e-20

			log_ll0 = np.log(llh_00 + 1e-11)
			log_ll1 = np.log(llh_01 + 1e-11)
			tolerance = np.abs(((log_ll0-log_ll1)/log_ll1))
			tolerance_history = np.append(tolerance_history[1:], [np.max(tolerance)])

			#errors.append(ds_error(n, K, self.alpha, self.mu, self.sigma, alpha_est, mu_est, sigma_est))
			likelihoods.append(np.sum(log_ll1)/N) 
			ll_history = np.append(ll_history[1:], [likelihoods[-1]-likelihoods[-2]])
			s_ests.append(s_est.copy())
			# if oscillations take place
			if np.where(ll_history < 0)[0].size >= 33 and ll_history[-1]>0:
				if beta != 1:
					print('ll break')
					break
				else:
					for k in range(K):
						mu_est[k] += np.random.normal()*sigma_est[k]**0.5

		print("Steps {} ".format(steps))
		beta_step.append((beta, steps-1))

		# print('Beta: {} --- alpha_est: {}, mu_est: {}, sigma_est: {}'.format(beta, alpha_ests, mu_est, sigma_est))
	return np.array(s_ests), steps, likelihoods

actual_likelihood = np.sum(np.log(noise_pdf(np.real(noise))*noise_pdf(np.imag(noise))))/N

p = np.exp(-0.01 * (np.arange(1,L+1) - 1)).reshape((L,1))
a = np.random.normal(0, 1, (L,1))
b = np.random.normal(0, 1, (L,1))

# initialization
s_est00 = ((a + 1j*b)*p)/(np.sum(p**2)) # same starting point for both

s_ests_daem, steps_daem, likelihoods_daem = solve(y, K=K, s_est=s_est00, thresh=1e-2, min_thresh=1e-4)
s_ests_em, steps_em, likelihoods_em = solve(y, K=K, thresh=1e-2, s_est=s_est00, min_thresh=1e-4, betas=[1])

plot_channel(s_ls, 'Least Squares')
plot_channel(s_ests_daem[-1], 'GMM-DAEM,steps='+str(steps_daem))
plot_channel(s_ests_em[-1], 'GMM-EM, steps='+str(steps_em))

plot_MSE(s_ests_daem, 'DAEM')
plot_MSE(s_ests_em, 'EM')

plt.figure('Likelihood')
plt.subplot(1,2,1)
plt.title(r'DAEM Likelihoods vs. Iterations, ')
plt.plot(likelihoods_daem)
plt.plot(np.repeat(actual_likelihood,len(likelihoods_daem)), label='Actual')	
plt.grid(True)
plt.subplot(1,2,2)
plt.title(r'EM Likelihoods vs. Iterations, ')
plt.plot(likelihoods_em)
plt.plot(np.repeat(actual_likelihood,len(likelihoods_em)), label='Actual')	
plt.grid(True)
plt.show()
