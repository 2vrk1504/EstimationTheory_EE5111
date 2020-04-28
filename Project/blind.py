import numpy as np 
import matplotlib.pyplot as plt

N = 512
L = 64
K = 4

SYMBOLS = np.array([1+1j, -1+1j, 1-1j, -1-1j])

n_sigma2 = 1

# channel parameters
c_sigma = 0.5 ** 0.5 	
lambda_ = 0.2
p = np.exp(-lambda_ * (np.arange(1,L+1) - 1)).reshape((L,1))
a = np.random.normal(0, c_sigma, (L,1))
b = np.random.normal(0, c_sigma, (L,1))
norm_p = np.sum(p**2)
# h0 is the true channel impulse response vector
h0 = ((a + 1j*b)*p)/norm_p	# h[k] = (a[k] + jb[k])p[k] / norm(p)

def plot_channel(hplot, label):
	plt.figure()
	plt.subplot(2,1,1)
	plt.title(label + ' Real and Imaginary')
	plt.grid(True)
	plt.stem([(hh[0]).real for hh in h0], linefmt='g-', markerfmt='go', label='Original', use_line_collection=True)
	plt.stem([(hh[0]).real for hh in hplot], linefmt='r:', markerfmt='ro', label=label, use_line_collection=True)
	plt.legend(loc='upper right')
	plt.subplot(2,1,2)
	plt.grid(True)
	plt.stem([(hh[0]).imag for hh in h0], linefmt='g-', markerfmt='go', label='Original', use_line_collection=True)
	plt.stem([(hh[0]).imag for hh in hplot], linefmt='r:', markerfmt='ro', label=label, use_line_collection=True)
	plt.legend(loc='upper right')


# Fourier Matrix
F = np.array([ [ np.exp((np.pi*2j*i*j)/N) for j in range(L)] for i in range(N)])
FH = F.conjugate().T

def P_gaussian(y, h, x, sigma2, beta):
	N = y.size
	val =  ((1/(2*np.pi*sigma2)**0.5)*np.exp((np.abs((x*F.dot(h) - y)).reshape(N)**2)/2/sigma2))**beta
	return val

def likelihood(y, h, nus, sigma2, K, beta):
	ll = 0
	for k in range(K):
		ll += (nus[k]**beta)*P_gaussian(y, h, SYMBOLS[k], sigma2, beta)
	return ll



def solve(y, K, L, betas=[0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 1], c_sigma_est=1, lambda_est=0, thresh=1e-10, max_steps=7000, history_length=100):

	N = y.size
	nu_est = np.array([1./K for j in range(K)])
	p = np.exp(-lambda_est * (np.arange(1,L+1) - 1)).reshape((L,1))
	a = np.random.normal(0, c_sigma_est, (L,1))
	b = np.random.normal(0, c_sigma_est, (L,1))
	h_est = ((a + 1j*b)*p)/(np.sum(p**2))

	plot_channel(h_est.copy(), 'lol')
	sigma2_est = np.sum(np.abs(y)**2)/N
	print('init', sigma2_est)

	likelihoods = []; beta_step = []; steps = 0;

	likelihoods.append(np.sum(np.log(likelihood(y, h_est, nu_est, sigma2_est, K, 1) + 1e-11))/N)
	
	for beta in betas:
		print('Maximization for beta = {}'.format(beta))

		if beta!=1:
			h_est += 1e-4*np.random.randn(L,1)

		llh_01 = likelihood(y, h_est, nu_est, sigma2_est, K, 1)
		llh_1 = likelihood(y, h_est, nu_est, sigma2_est, K, beta)

		# define h[k, i] = probability that xi belongs to class k
		# however, the following is being done for numerical stability
		q = np.array([(nu_est[k]**beta)*P_gaussian(y, h_est, SYMBOLS[k], sigma2_est, beta)/(llh_1+1e-9) for k in range(K-1)])
		q = np.append(q, [(1 - np.sum(q, axis=0))], axis=0)
		
		tolerance = np.ones(N)
		tolerance_history = np.ones(history_length)
		ll_history = np.ones(history_length)

		if beta == 1:
			thresh = 1e-10
			# tolerance_history_thresh = 1e-7

		while tolerance_history[-1] >= thresh and steps <= max_steps:
			steps += 1
			print("Step {}".format(steps), end='\r')
			llh_00 = llh_01.copy()
			llh_0 = llh_1.copy()

			temp = np.zeros((N,1), dtype=np.complex128)
			temp1 = np.zeros((N,1), dtype=np.complex128)
			for k in range(K):
				temp += np.conjugate(SYMBOLS[k])*y*(q[k].reshape((N, 1)))
				temp1 += (np.abs(SYMBOLS[k])**2)*(q[k].reshape((N,1)))
				nu_est[k] = np.sum(q[k])/N

			h_est = FH.dot(temp/temp1)/N

			for k in range(K):
				sigma2_est += np.sum((np.abs(SYMBOLS[k]*F.dot(h_est)-y)**2)*q[k].reshape((N,1)))
			sigma2_est /= N

			llh_01 = likelihood(y, h_est, nu_est, sigma2_est, K, 1)
			llh_1 = likelihood(y, h_est, nu_est, sigma2_est, K, beta)

			# The following is being done for numerical stability
			q = np.array([(nu_est[k]**beta)*P_gaussian(y, h_est, SYMBOLS[k], sigma2_est, beta)/(llh_1+1e-9) for k in range(K-1)])
			q = np.append(q, [(1 - np.sum(q, axis=0))], axis=0)

			log_ll0 = np.log(llh_00 + 1e-11)
			log_ll1 = np.log(llh_01 + 1e-11)
			tolerance = np.abs(((log_ll0-log_ll1)/log_ll1))
			tolerance_history = np.append(tolerance_history[1:], [np.max(tolerance)])

			likelihoods.append(np.sum(log_ll1)/N) 
			ll_history = np.append(ll_history[1:], [likelihoods[-1]-likelihoods[-2]])

			# if oscillations take place
			# if np.where(ll_history < 0)[0].size >= 33 and ll_history[-1]>0:
			# 	if beta != 1:
			# 		print('ll break')
			# 		break
			# 	else:
			# 		h_est += 1e-4*np.random.randn(L,1)

		print("Steps {} ".format(steps))
		beta_step.append((beta, steps-1))

		# print('Beta: {} --- alpha_est: {}, mu_est: {}, sigma_est: {}'.format(beta, alpha_ests, mu_est, sigma_est))
	return h_est, sigma2_est, nu_est, beta_step, likelihoods

X = np.array([SYMBOLS[int(np.random.rand()*K)] for i in range(N)]).reshape((N,1))
y = X*(F.dot(h0)) + np.random.normal(0, n_sigma2**0.5, (N,1)) # process noise


h_est, sigma2_est, nu_est, beta_step, likelihoods = solve(y, K, L)
print('sigma2_est', sigma2_est)
print('nu_est', nu_est)


plt.figure('Likelihood')
plt.subplot(1,2,1)
plt.title(r'Likelihoods vs. Iterations, ')#$(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
plt.plot(likelihoods)
# plt.plot(np.repeat(actual_likelihood_daem,len(likelihoods_daem)), label='Actual')	
plt.grid(True)


plot_channel(h_est, 'Estimated')
plt.show()
