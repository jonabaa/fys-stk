from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_log_error, mean_absolute_error

# creating sample data
x = 2*np.random.rand(100,1)
y = 4+3*x+np.random.randn(100,1)

# fitting function/computing beta
xb = np.c_[np.ones((100,1)), x]
beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)
xnew = np.array([[0],[2]])

xbnew = np.c_[np.ones((100,1)), x]

ypredict = xbnew.dot(beta)

# fitting function with scikit-learn
linreg = LinearRegression()
linreg.fit(x,y)
ypredictSL = linreg.predict(x)

# print the mean square error
print("MSE of your fit: %.4f" % mean_squared_error(y, ypredict))
print("MSE of scikit fit: %.4f" % mean_squared_error(y, ypredictSL))

# print the R2-score
print('Variance score: %.2f' % r2_score(y, ypredict))

# plotting
plt.plot(x, ypredict, "r-")
plt.plot(x, ypredictSL, "b--")
plt.plot(x, y ,'ro')
plt.axis([0,2.0,0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Linear Regression')
plt.show()

