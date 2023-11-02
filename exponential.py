import numpy as np

lambda_param = 1.5
sample_size = 50

# Генерация выборки
sample = np.random.exponential(scale=1/lambda_param, size=sample_size)

# Вывод выборки
print(sample)
