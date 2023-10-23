import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

mu = 12
sigma = 0.5

x = np.random.normal(mu, sigma, 50)
x.sort()

plt.plot (x, sts.norm.pdf(x, mu, sigma), label='μ:0, σ: 2')
plt.show()