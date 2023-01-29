from sklearn.linear_model import LinearRegression

def regressor(Y, X):
    lr = LinearRegression()
    lr.fit(X,Y)
    return lr.coef_

reg1_env1 = regressor(Ye1,X1e1)[0]
reg2_env1 = regressor(Ye1,X2e1)[0]
reg3_env1 = regressor(Ye1,np.concatenate([X1e1,X2e1], axis=1))[0]

reg1_env2 = regressor(Ye2,X1e2)[0]
reg2_env2 = regressor(Ye2,X2e2)[0]
reg3_env2 = regressor(Ye2,np.concatenate([X1e2,X2e2], axis=1))[0]
                      
print("By running such regressions (despite different environnements), one might naively expect to \nhave quite the same result for each pair of regressions on both environnement but this is not \nthe case. \nFor the first regression we obtain %f as coefficient on environnement 1 and %f \non environnement 2.\nFor the second regression we obtain %f as coefficient on environnement 1 and %f \non environnement 2.\nFor the third regression we obtain the couple (%f, %f) as coefficients on \nenvironnement 1 and (%f, %f) on environnement 2." % (reg1_env1, reg1_env2, reg2_env1, reg2_env2, reg3_env1[0], reg3_env1[1], reg3_env2[0], reg3_env2[1]))