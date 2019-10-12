import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# initialize seaborn
sns.set()

# Generate and shape the training data
randomNumGenObj = np.random.RandomState(42)  # this initializes a Mersenne Twister object i.e. with different methods to generate random numbers
xtrain = 10 * randomNumGenObj.rand(50)
ytrain = 2*xtrain - 1 + randomNumGenObj.randn(50)

Xtrain = xtrain[:, np.newaxis]  # reshape to ensure it is a matrix i.e. using np.newaxis to add a column
print(Xtrain.shape)

# Plot training dataset
plt.scatter(xtrain, ytrain)
plt.show()

# Generate and shape the validation data
xtest = np.linspace(-1, 11)
ytest = 2*xtest - 1 + randomNumGenObj.randn(50)
Xtest = xtest[:, np.newaxis]

# Apply modelling class
model = LinearRegression(fit_intercept=True)  # initialize
model.fit(Xtrain, ytrain)  # fit to data
print("Fitted Coefficient: {}".format(model.coef_))  # print the value of the fitted coefficient
print("Fitted Intercept: {}".format(model.intercept_))  # print the value of the fitted intercept


# Test the trained model with new data
y_predicted = model.predict(Xtest)

# Plot testing dataset
plt.scatter(xtrain, ytrain)
plt.plot(xtest, y_predicted)
plt.show()