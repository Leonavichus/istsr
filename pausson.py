import numpy as np

lambda_param = 3.5
sample_size = 50

# Генерация выборки
sample = np.random.poisson(lambda_param, sample_size)

# Вывод выборки
print(sample)