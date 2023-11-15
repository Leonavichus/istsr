import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

mu = 12
sigma = 2.7
sample_size = 50

# Генерация выборки
sample = np.random.normal(mu, sigma, sample_size)

# Вывод выборки
print(sample)

# Описательная статистика
print("Описательная статистика:")
print("Среднее:", np.mean(sample))
print("Стандартная ошибка:", np.std(sample) / np.sqrt(len(sample)))
print("Медиана:", np.median(sample))
print("Мода:", stats.mode(sample))
print("Стандартное отклонение:", np.std(sample))
print("Дисперсия выборки:", np.var(sample))
print("Эксцесс:", stats.kurtosis(sample))
print("Асимметричность:", stats.skew(sample))
print("Интервал:", )
print("Минимум:", np.amin(sample))
print("Максимум:", np.amax(sample))
print("Сумма:", np.sum(sample))
print("Счет:", len(sample))
print("Уровень надежности:", )

# Гистограмма
sns.histplot(sample, kde=True)
plt.title('Гистограмма')
plt.show()
