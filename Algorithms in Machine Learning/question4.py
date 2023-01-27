import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

X1, X2, Y = sample(sigma=10, n=10000)

Dlin = []
C =  np.linspace(-1, 1, 101)
w  = np.array([1,0])

for c in C:
    X=np.concatenate([X1,X2], axis=1)
    phi = np.array([[1, 0],
                   [0, c]])
    u = (phi.T @ X.T @ X @ phi @ w - Y.T @ X @ phi)/Y.shape[0]
    Dlin.append((u @ u.T).sum())
    #Dlin.append((np.linalg.norm(phi.T @ X.T @ X @ phi @ w - phi.T @ X.T @ Y) / X.shape[0])**2)

with plt.xkcd(0.8, 40, 5): #don't hesitate to comment this line if you don't enjoy the graph
    fig, ax = plt.subplots(figsize=(8,5))
    ax.annotate(
        'I did not lie, that is\n much more smooth',
        xy=(0, 0), arrowprops=dict(arrowstyle='->'), xytext=(-0.5, 6000))
    ax.plot(C, Dlin)
    plt.ylabel('Ddist')
    plt.xlabel('c')
    plt.title('Influence of the variable c on Dlin')
    plt.show()