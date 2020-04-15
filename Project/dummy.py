import numpy as np
import matplotlib.pyplot as plt

def toss(alpha):
	x = np.random.random()
	last = 0
	for k in range(len(alpha)):
		if x >= last and x<last+alpha[k]:
			return k
		else:
			last += alpha[k]



mu = np.array([
		np.array([[0],
				  [0]]),
		np.array([[10],
				  [0]]),
		])
sigma = np.array([
			np.array([[0.1, .099],
					  [.099, .1]]),
			np.array([[1, 0],
					  [0, 1]]),
		])

alpha = [0.9, 0.1]

ws=[]; vs=[];
for k in range(len(alpha)):
	w, v = np.linalg.eig(sigma[k])
	ws.append(w); vs.append(v)

print(ws)
print(vs)

color = ['blue', 'green']

plt.figure()
plt.grid(True)
X = np.empty((2,0))

Xc= [np.empty((2,0)), np.empty((2,0))]
for j in range(1000):
	kc = toss(alpha)
	muc = mu[kc]; w = ws[kc]; v=vs[kc] 
	Xc[kc] = np.append(Xc[kc], muc + v.dot((w**0.5).reshape(2,1)*(np.random.randn(2,1))), axis=1)

for k in range(len(alpha)):
	plt.plot(Xc[k][0], Xc[k][1], '.', color=color[k])


# Y = np.random.multivariate_normal(mu.reshape(2), sigma, size=1000)
# plt.plot(Y[:,0], Y[:,1], 'r.')

theta = np.linspace(0, 2*np.pi, 100)
for k in range(len(alpha)):
	w = ws[k]; v = vs[k]
	xx0 = 3*(w[0]**0.5)*np.cos(theta) 
	xx1 = 3*(w[1]**0.5)*np.sin(theta) 
	x00 = v[0][0]*xx0 + v[0][1]*xx1 + mu[k][0][0]
	x11 = v[1][0]*xx0 + v[1][1]*xx1 + mu[k][1][0]
	plt.plot(x00, x11)



plt.show()