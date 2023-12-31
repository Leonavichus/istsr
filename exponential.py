import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

lambda_param = 1.5
sample_size = 50

# Генерация выборки
sample = np.random.exponential(scale=1/lambda_param, size=sample_size)

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

# Оценка параметров показательного распределения
amin = np.amin(sample)
amax = np.amax(sample)
avg = np.mean(sample)

# Оценка mu
est_mu = 1 / avg
print('Оценка mu:', est_mu)

# Шаг
step = (amax - amin) / 6
print('Шаг:', step)

# Формула Стерджа
formula_sturge = 1 + 1.4 * np.log(len(sample))
print('Формула Стерджа:', formula_sturge)

# Создаем массив и заполняем его значениями
limits = np.arange(0, amax + step, step)

# Инициализация массива для хранения количества чисел между элементами
counts = []

# Учет чисел между нулем и первым элементом второго массива
selected_elements = sample[(sample >= 0) & (sample <= limits[0])]
counts.append(len(selected_elements))

# Перебор элементов второго массива
for i in range(len(limits) - 1):
    # Выбор элементов из первого массива, которые лежат между соответствующими элементами второго массива
    selected_elements = sample[(sample > limits[i])
                               & (sample <= limits[i + 1])]

    # Запись количества выбранных элементов в массив
    counts.append(len(selected_elements))

x1 = limits[:-1]
x2 = limits[1:]

fx1 = [1 - np.exp(-est_mu * x) for x in x1]
fx2 = [1 - np.exp(-est_mu * x) for x in x2]
p = np.array(fx2) - np.array(fx1)
theor_frequency = [x * len(sample) for x in p]

df = pd.DataFrame({
    'x(i)': list(x1),
    'x(i+1)': list(x2),
    'Частота эмпир': list(counts[1:]),
    'F(x(i))': list(fx1),
    'F(x(i+1))': list(fx2),
    'P': list(p),
    'Частота теор': list(theor_frequency),
})

print(df)

print('Сумма:', sum(theor_frequency))


def cs(n, y):
    return stats.chisquare(n, np.sum(n)/np.sum(y) * y)


observed = np.array(counts[1:])
expected = np.array(theor_frequency)

chi2, p_value = cs(observed, expected)

# Вывод результатов
print(f'Хи-квадрат: {chi2}')
print(f'p-значение: {p_value}')

# Проверка гипотезы
alpha = 0.05
print(f'Уровень значимости: {alpha}')
print(
    f'Гипотеза о экспоненциальном распределении: {"принимается" if p_value > alpha else "отвергается"}')

statistic, p_value2 = stats.f_oneway(observed, expected)

# Вывод результатов
print("F-статистика:", statistic)
print("p-значение:", p_value2)

# Проверка наличия статистической значимости
print(
    f'Гипотеза о экспоненциальном распределении: {"принимается" if p_value2 > alpha else "отвергается"}')
