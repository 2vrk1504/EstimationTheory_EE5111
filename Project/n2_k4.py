import numpy as np 
import matplotlib.pyplot as plt 
from algos import Solver

print('START')
N = 1000		# number of samples
K = 4			# number of mixed Gaussians
n = 2			# dimension

# 4 cluster case
mu=np.array([[[2],[1]],[[-2],[1]],[[2],[-1]],[[-2],[-1]]])
#sigma=np.array([0.5*np.identity(2) for j in range(K)])/100

sigma = np.array([		# covariance matrices
	np.array([[5, 0.5],
			  [0.5, 1]]), 
	np.array([[1, 0.5],
			  [0.5, 6]]),
	np.array([[14, 0.5],
			  [0.5, 1]]), 
	np.array([[1, 0.5],
			  [0.5, 1]])
])/100

alpha=[0.05,0.5,0.15,0.3] #np.linspace(0.6, 0.6, 1) # mixing coefficients

# 2 Clusters case
'''
mu=np.array([[[2],[1]],[[-2],[1]]])
sigma = np.array([		# covariance matrices
	np.array([[1, 0.5],
			  [0.5, 1]]), 
	np.array([[1, 0.5],
			  [0.5, 1]])])/100

alpha=[0.1,0.9]
'''
# Data Generation

l=[]
X=[]
for i in range(N):

	z=np.random.choice(np.arange(0, K), p=alpha)
	l.append(z)
	X.append(np.random.multivariate_normal(mu[z].T[0],sigma[z]))
X=np.array(X).T



colors = ['red', 'pink', 'brown','gold']

errorss_daem = []; alphass_daem = []; muss_daem = [];
errorss_em = []; alphass_em = []; muss_em = []
bs = []


solver = Solver(alpha=np.array(alpha), mu=mu, sigma=sigma)


#Data generation



print('Actual:')
print('alpha: {}\nmu:\n{}\nsigma:\n{}\n\n'.format(alpha, mu, sigma))
alpha_est_daem, mu_est_daem, sigma_est_daem, errors_daem, steps, beta_step, likelihoods_daem, actual_likelihood_daem = solver.DAEM_GMM(X=X, thresh=1e-10, K=K, max_steps=5000)
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

plt.figure('Distribution')
plt.title(r'Distribution and prediction')#$(\mu_1,\mu_2)=($'+str(-mu_iter)+','+str(mu_iter)+')')

plt.scatter(X[0],X[1],marker='.')

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

# plt.legend(loc='upper right')
plt.grid(True)
plt.show()
