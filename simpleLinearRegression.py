import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate the data
randomNumGenObj = np.random.RandomState(42)  # this initializes a Mersenne Twister object i.e. with different methods to generate random numbers
x = 10 * randomNumGenObj.rand(50)
y = 2*x - 1 + randomNumGenObj.randn(50)

# initialize seaborn
sns.set()

# Plot current dataset
plt.scatter(x,y)
