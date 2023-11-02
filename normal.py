import numpy as np

mu = 12
sigma = 2.7
sample_size = 50

# Генерация выборки
sample = np.random.normal(mu, sigma, sample_size)

# Вывод выборки
print(sample)
