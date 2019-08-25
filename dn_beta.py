"""Диаграмма направленности в азимутальной плоскости."""
import radar
import numpy as np
import matplotlib.pyplot as plt
import time

# Начальное время выполнения программы
tic = time.clock()

# Исходные данные

# Количество столбцов АР
n_x = 30
# Расстояние между столбцами АР, м
d_x = 0.1
# Частота, Гц
f = 1000e6
# Начальный угол сканирования, угл. градус
beta_start = -30.0
# Шаг сканирования, угл. градус
beta_step = 0.1
# Конечный угол сканирования, угл. градус
beta_finish = 30.0
# Массив азимутов, угл. градус
beta = np.arange(beta_start, beta_finish + beta_step, beta_step, dtype=float)
# Угол, на который установлена АР, угл. градус
beta_0 = 0.0
# Диаграмма направленности элемента (косинус)
dn_element = lambda angle: np.cos(angle)

# ДН по азимуту
dn_beta = radar.dn_beta(n_x, d_x, f, beta, beta_0, dn_element)

# График ДН в азимутальной плоскости
plt.figure()
plt.title("Диаграмма направленности в азимутальной плоскости")
plt.xlabel("Азимут, угл. градус")
# Если plot_linear - линейный масштаб, иначе логарифмический
plot_linear = True
if plot_linear == True:
    plt.plot(beta, dn_beta)
    plt.ylabel("Амплитуда")
else:
    plt.plot(beta, 20 * np.log10(dn_beta))
    plt.ylabel("Амплитуда, дБ")
plt.grid()
plt.show()

# Ширина ДН на уровне -3 дБ. Чтобы получить более точный результат,
# необходимо шаг сканирования сделать чаще
delta_beta = radar.delta_dn(beta, dn_beta)
print(delta_beta)

# Время выполнения программы, с
toc = time.clock() - tic