import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

sample1 = np.array([12, 20, 18, 18, 12, 17, 11, 24, 15, 13, 15, 4, 13,
                    19, 16, 8, 11, 22, 16, 15, 11, 7, 14, 16, 18, 16, 14, 19, 13, 16])

sample2 = np.array([8, 14, 14, 14, 13, 16, 14, 24, 11, 15, 20, 17, 8,
                    13, 17, 11, 17, 11, 21, 16, 18, 13, 16, 13, 19, 14, 23, 18, 14, 17])

# Описательная статистика
print('Описательная статистика первой выборки:')
print('Среднее:', np.mean(sample1))
print('Стандартная ошибка:', np.std(sample1) / np.sqrt(len(sample1)))
print('Медиана:', np.median(sample1))
print('Мода:', stats.mode(sample1))
print('Стандартное отклонение:', np.std(sample1))
print('Дисперсия выборки:', np.var(sample1))
print('Эксцесс:', stats.kurtosis(sample1))
print('Асимметричность:', stats.skew(sample1))
print('Доверительный интервал для математического ожидания (95,0%):', stats.t.interval(confidence=0.95, df=len(
    sample1)-1, loc=np.mean(sample1), scale=stats.sem(sample1)))
print('Минимум:', np.amin(sample1))
print('Максимум:', np.amax(sample1))
print('Сумма:', np.sum(sample1))
print('Счет:', len(sample1))
print('Доверительный интервал для дисперсии (95,0%):',  stats.t.interval(confidence=0.95, df=len(sample1)-1,
      loc=np.var(sample1), scale=2*np.var(sample1)/len(sample1)))

# Описательная статистика
print('Описательная статистика второй выборки:')
print('Среднее:', np.mean(sample2))
print('Стандартная ошибка:', np.std(sample2) / np.sqrt(len(sample2)))
print('Медиана:', np.median(sample2))
print('Мода:', stats.mode(sample2))
print('Стандартное отклонение:', np.std(sample2))
print('Дисперсия выборки:', np.var(sample2))
print('Эксцесс:', stats.kurtosis(sample2))
print('Асимметричность:', stats.skew(sample2))
print('Доверительный интервал для математического ожидания (95,0%):', stats.t.interval(confidence=0.95, df=len(
    sample2)-1, loc=np.mean(sample2), scale=stats.sem(sample2)))
print('Минимум:', np.amin(sample2))
print('Максимум:', np.amax(sample2))
print('Сумма:', np.sum(sample2))
print('Счет:', len(sample2))
print('Доверительный интервал для дисперсии (95,0%):',  stats.t.interval(confidence=0.95, df=len(sample2)-1,
      loc=np.var(sample2), scale=2*np.var(sample2)/len(sample2)))

# Гистограмма
plt.subplot(1, 2, 1)
sns.histplot(sample1, kde=True)
plt.title('Гистограмма')
plt.xlabel('Значение')
plt.ylabel('Частота')

plt.subplot(1, 2, 2)
sns.histplot(sample2, kde=True)
plt.title('Гистограмма')
plt.xlabel('Значение')
plt.ylabel('Частота')

# Полигон
plt.subplot(1, 2, 1)
sns.kdeplot(sample1, cumulative=False, fill=True)
plt.title('Полигон')
plt.xlabel('Значение')
plt.ylabel('Плотность')

plt.subplot(1, 2, 2)
sns.kdeplot(sample2, cumulative=False, fill=True)
plt.title('Полигон')
plt.xlabel('Значение')
plt.ylabel('Плотность')

# Кумулянта
plt.subplot(1, 2, 1)
sns.kdeplot(sample1, cumulative=True)
plt.title('Кумулянта')
plt.xlabel('Значение')
plt.ylabel('Вероятность')

plt.subplot(1, 2, 2)
sns.kdeplot(sample2, cumulative=True)
plt.title('Кумулянта')
plt.xlabel('Значение')
plt.ylabel('Вероятность')

# Ящик с усами
plt.subplot(1, 2, 1)
sns.boxplot(sample1)
plt.title('Ящик с усами')
plt.xlabel('Значение')

plt.subplot(1, 2, 2)
sns.boxplot(sample2)
plt.title('Ящик с усами')
plt.xlabel('Значение')

# Q-Q Plot norm
plt.subplot(2, 2, 1)
stats.probplot(sample1, dist='norm', plot=plt)
plt.title('Q-Q Plot-norm')
plt.xlabel('Теоретические квантили')
plt.ylabel('Выборочные квантили')

plt.subplot(2, 2, 3)
stats.probplot(sample2, dist='norm', plot=plt)
plt.title('Q-Q Plot-norm')
plt.xlabel('Теоретические квантили')
plt.ylabel('Выборочные квантили')

# Q-Q Plot expon
plt.subplot(2, 2, 2)
stats.probplot(sample1, dist='expon', plot=plt)
plt.title('Q-Q Plot-expon')
plt.xlabel('Теоретические квантили')
plt.ylabel('Выборочные квантили')

plt.subplot(2, 2, 4)
stats.probplot(sample2, dist='expon', plot=plt)
plt.title('Q-Q Plot-expon')
plt.xlabel('Теоретические квантили')
plt.ylabel('Выборочные квантили')
