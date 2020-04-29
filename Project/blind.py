import numpy as np 
import matplotlib.pyplot as plt

N = 512
L = 2
K = 4

SYMBOLS = np.array([1+1j, -1+1j, 1-1j, -1-1j])

n_sigma2 = 0.1

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
	dist2 = np.abs((x*(F.dot(h)) - y)).reshape(N)**2
	val = ((1/(2*np.pi*sigma2)**0.5)*np.exp(-dist2/2/sigma2))**beta
	return val

def likelihood(y, h, nus, sigma2, K, beta):
	ll = 0
	for k in range(K):
		ll += (nus[k]**beta)*P_gaussian(y, h, SYMBOLS[k], sigma2, beta)
	return ll



def solve(y, K, L, betas=[0.6, 0.8, 1], c_sigma_est=1, lambda_est=0, thresh=1e-10, max_steps=5000, tolerance_history_thresh=1e-2, history_length=50):

	N = y.size
	nu_est = np.array([1./K for j in range(K)])
	p = np.exp(-lambda_est * (np.arange(1,L+1) - 1)).reshape((L,1))
	a = np.random.normal(0, c_sigma_est, (L,1))
	b = np.random.normal(0, c_sigma_est, (L,1))

	# initialization
	h_est = ((a + 1j*b)*p)/(np.sum(p**2))

	sigma2_est = np.sum(np.abs(y)**2)/N
	print('init', sigma2_est)

	likelihoods = []; beta_step = []; steps = 0;
	h_ests = [h_est.copy()]

	actual_likelihood = np.sum(np.log(likelihood(y, h0, np.array([0.25, 0.25, 0.25, 0.25]) , n_sigma2, K, 1) + 1e-11))/N
	likelihoods.append(np.sum(np.log(likelihood(y, h_est, nu_est, sigma2_est, K, 1) + 1e-11))/N)
	
	for beta in betas:
		print('Maximization for beta = {}'.format(beta))

		if beta!=1:
			h_est += 1*np.random.randn(L,1)

		llh_01 = likelihood(y, h_est, nu_est, sigma2_est, K, 1)
		llh_1 = likelihood(y, h_est, nu_est, sigma2_est, K, beta)

		# define h[k, i] = probability that xi belongs to class k
		# however, the following is being done for numerical stability
		q = np.array([(nu_est[k]**beta)*P_gaussian(y, h_est, SYMBOLS[k], sigma2_est, beta)/(llh_1+1e-9) for k in range(K-1)])
		q = np.append(q, [(1 - np.sum(q, axis=0))], axis=0)
		
		tolerance_history = np.ones(history_length)
		ll_history = np.ones(history_length)

		if beta == 1:
			thresh = 1e-10
			tolerance_history_thresh = 1e-7

		while tolerance_history[-1] >= thresh and steps <= max_steps:
			steps += 1
			print("Step {}".format(steps), end='\r')
			llh_00 = llh_01.copy()
			llh_0 = llh_1.copy()

			temp = np.zeros((N,1), dtype=np.complex128)
			temp1 = np.zeros((N,1), dtype=np.complex128)
			for k in range(K):
				temp += np.conjugate(SYMBOLS[k])*y*(q[k].reshape((N, 1)))
				temp1 += (np.abs(SYMBOLS[k])**2)*(q[k].reshape((N,1)) + 1e-9)
				nu_est[k] = np.sum(q[k])/N

			# print('temp', temp)
			# print('temp1', temp1)

			mat = np.linalg.inv(FH.dot(np.diag(temp1.reshape(N)).dot(F)))
			h_est = mat.dot(FH.dot(temp))

			for k in range(K):
				sigma2_est += np.sum((np.abs(SYMBOLS[k]*F.dot(h_est)-y)**2)*q[k].reshape((N,1)))
			sigma2_est /= N

			llh_01 = likelihood(y, h_est, nu_est, sigma2_est, K, 1)
			llh_1 = likelihood(y, h_est, nu_est, sigma2_est, K, beta)

			if np.max(tolerance_history) <= tolerance_history_thresh:
				h_est += 1*np.random.randn(L,1)

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
			if np.where(ll_history < 0)[0].size >= 33 and ll_history[-1]>0:
				if beta != 1:
					print('ll break')
					break
				else:
					h_est += 1e-3*np.random.randn(L,1)
			h_ests.append(h_est.copy())

		print("Steps {} ".format(steps))
		beta_step.append((beta, steps-1))
		# print('Beta: {} --- alpha_est: {}, mu_est: {}, sigma_est: {}'.format(beta, alpha_ests, mu_est, sigma_est))
	return h_ests, sigma2_est, nu_est, beta_step, likelihoods, actual_likelihood

X = np.array([SYMBOLS[int(np.random.rand()*K)] for i in range(N)]).reshape((N,1))
y = X*(F.dot(h0)) + (0.5**0.5)*(np.random.normal(0, n_sigma2**0.5, (N,1)) + 1j*np.random.normal(0, n_sigma2**0.5, (N,1))) # process noise


h_ests, sigma2_est, nu_est, beta_step, likelihoods, actual_likelihood = solve(y, K, L)
print('sigma2_est', sigma2_est)
print('nu_est', nu_est)
print('h_est', h_ests[-1])

plt.figure('Y')
plt.title('Y')
plt.plot(np.real(y), np.imag(y), 'b.', label='Y')
plt.plot(np.real(X), np.imag(X), 'ro', label='X')
Z = y/(F.dot(h0))
plt.plot(np.real(Z), np.imag(Z), 'g.', label='Z')
plt.grid(True)
plt.legend(loc='upper right')



plt.figure('Likelihood')
plt.title(r'Likelihoods vs. Iterations, ')#$(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')
plt.plot(likelihoods)
plt.plot(np.repeat(actual_likelihood,len(likelihoods)), label='Actual')	
plt.grid(True)

plot_channel(h_ests[-1], 'Est')

plt.show()
