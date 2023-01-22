import numpy as np

#Generate n samples under environnement sigma
def sample(sigma, n):
    X1 = np.random.normal(0,sigma,n).reshape(-1,1)
    Y = X1 + np.random.normal(0,sigma,n).reshape(-1,1)
    X2 = Y + np.random.normal(0,1,n).reshape(-1,1)
    return X1, X2, Y

n = 10000 # number of samples for each environnement 

sigma1 = 10 # variance for the environnement 1 noise 
sigma2 = 0.1 # variance for the environnement 2 noise 

X1e1, X2e1, Ye1 = sample(sigma1, n)
X1e2, X2e2, Ye2 = sample(sigma2, n)