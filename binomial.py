import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

m = 5
p = 0.9
sample_size = 50

# Генерация выборки
sample = np.random.binomial(n=m, p=p, size=sample_size)

print(sample)

# Часть 1

# Описательная статистика
print('Описательная статистика:')
print('Среднее:', np.mean(sample))
print('Стандартная ошибка:', np.std(sample) / np.sqrt(len(sample)))
print('Медиана:', np.median(sample))
print('Мода:', stats.mode(sample))
print('Стандартное отклонение:', np.std(sample))
print('Дисперсия выборки:', np.var(sample))
print('Эксцесс:', stats.kurtosis(sample))
print('Асимметричность:', stats.skew(sample))
print('Доверительный интервал для математического ожидания (95,0%):', stats.t.interval(confidence=0.95, df=len(
    sample)-1, loc=np.mean(sample), scale=stats.sem(sample)))
print('Минимум:', np.amin(sample))
print('Максимум:', np.amax(sample))
print('Сумма:', np.sum(sample))
print('Счет:', len(sample))
print('Доверительный интервал для дисперсии (95,0%):',  stats.t.interval(confidence=0.95, df=len(sample)-1,
      loc=np.var(sample), scale=2*np.var(sample)/len(sample)))

print()

plt.figure(figsize=(14, 10))

# Гистограмма
plt.subplot(3, 3, 1)
sns.histplot(sample, kde=True)
plt.title('Гистограмма')
plt.xlabel('Значение')
plt.ylabel('Частота')

# Полигон
plt.subplot(3, 3, 2)
sns.kdeplot(sample, cumulative=False, fill=True)
plt.title('Полигон')
plt.xlabel('Значение')
plt.ylabel('Плотность')

# Кумулянта
plt.subplot(3, 3, 3)
sns.kdeplot(sample, cumulative=True)
plt.title('Кумулянта')
plt.xlabel('Значение')
plt.ylabel('Вероятность')

# Ящик с усами
plt.subplot(3, 3, 4)
sns.boxplot(sample)
plt.title('Ящик с усами')
plt.xlabel('Значение')

# Q-Q Plot norm
plt.subplot(3, 3, 5)
stats.probplot(sample, dist='norm', plot=plt)
plt.title('Q-Q Plot-norm')
plt.xlabel('Теоретические квантили')
plt.ylabel('Выборочные квантили')

# Q-Q Plot expon
plt.subplot(3, 3, 6)
stats.probplot(sample, dist='expon', plot=plt)
plt.title('Q-Q Plot-expon')
plt.xlabel('Теоретические квантили')
plt.ylabel('Выборочные квантили')

# Эмпирическая функция распределения
plt.subplot(3, 3, 7)
sns.ecdfplot(sample)
plt.title('Эмпирическая функция распределения')
plt.xlabel('Значение')
plt.ylabel('Вероятность')

plt.tight_layout()
plt.show()

# Часть 2
print()

# Оценка параметров биномиального распределения
amin = np.amin(sample)
amax = np.amax(sample)
avg = np.mean(sample)

# Оценка p
est_p = avg / m
print('Оценка p:', est_p)

# Используйте функцию unique() для получения уникальных элементов и их частот
unique_elements, counts = np.unique(sample, return_counts=True)

# Создайте словарь, чтобы связать уникальные элементы с их частотами
frequency_dict = dict(zip(unique_elements, counts))

print()

probability = []
for element, frequency in frequency_dict.items():
    probability.append(stats.binom.pmf(element, m, est_p))

theor_frequency = [x * len(sample) for x in probability]

df = pd.DataFrame({
    'Значение': list(frequency_dict.keys()),
    'Частота': list(frequency_dict.values()),
    'Вероятность': list(probability),
    'Теоретическая частота': list(theor_frequency),
})

print(df)

print('Сумма:', sum(theor_frequency))


def cs(n, y):
    return stats.chisquare(n, np.sum(n)/np.sum(y) * y)


observed = np.array(list(frequency_dict.values()))
expected = np.array(theor_frequency)

chi2, p_value = cs(observed, expected)

# Вывод результатов
print(f'Хи-квадрат: {chi2}')
print(f'p-значение: {p_value}')

# Проверка гипотезы
alpha = 0.05
print(f'Уровень значимости: {alpha}')
print(
    f'Гипотеза о биномиальном распределении: {"принимается" if p_value > alpha else "отвергается"}')

statistic, p_value2 = stats.f_oneway(observed, expected)

# Вывод результатов
print("F-статистика:", statistic)
print("p-значение:", p_value2)

# Проверка наличия статистической значимости
print(
    f'Гипотеза о биномиальном распределении: {"принимается" if p_value2 > alpha else "отвергается"}')
