import matplotlib.pyplot as plt
import numpy as np

X1, X2, Y = sample(sigma=10, n=10000)

Ddist = []
C =  np.linspace(-1, 1, 101)
w  = np.array([1,0])

for c in C:
    X=np.concatenate([X1,X2], axis=1)
    phi = np.array([[1, 0],
                   [0, c]])
    we = regressor(Y, X@phi)
    Ddist.append(((we-w)@(we-w).T).sum())
plt.figure(figsize=(8,5))
plt.ylim(-1, 20)
plt.plot(C, Ddist)
plt.ylabel('Ddist')
plt.xlabel('c')
plt.title('Influence of the variable c on the Ddist')
plt.show()