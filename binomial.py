import numpy as np

m = 5
p = 0.9
sample_size = 50

# Генерация выборки
sample = np.random.binomial(m, p, sample_size)

# Вывод выборки
print(sample)
