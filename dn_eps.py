"""Диаграмма направленности в угломестной плоскости."""
import radar
import numpy as np
import matplotlib.pyplot as plt
import time

# Начальное время выполнения программы
tic = time.perf_counter()

# Исходные данные

# Количество строк АР
n_y = 30
# Расстояние между строками АР, м
d_y = 0.1
# Частота, Гц
freq = 1000e6
# Угол наклона антенной решётки, угл. градус
eps_a = 10
# Высота подъёма геометрического центра антенны, м
h_gc = 10
# Начальный угол сканирования, угл. градус
eps_start = 0.1
# Шаг сканирования, угл. градус
eps_step = 0.1
# Конечный угол сканирования, угол. градус
eps_finish = 30.0
# Массив углов места цели, угл. градус
eps = np.arange(eps_start, eps_finish + eps_step, eps_step, dtype=float)
# Угол, на который настроен пространственный фильтр, угл. градус
eps_lou = 0.1
# Диаграмма направленности элемента (косинус)
dn_element = lambda angle: np.cos(angle)
# Наличие отражений от земной поверхности
refraction = True
# Глубина шероховатости, м (в дециметровом диапазоне присутствует)
h_sher = 1.5 * radar.c / freq

# Диаграмма направленности в угломестной плоскости
dn_eps = radar.dn_eps(n_y, d_y, freq, eps_a, eps, eps_lou, dn_element, h_gc,
                      refraction=refraction, h_sher=h_sher)

# График ДН в угломестной плоскости
plt.figure()
plt.title("Диаграмма направленности в угломестной плоскости")
plt.xlabel("Угол места, угл. градус")
# Если plot_linear - линейный масштаб, иначе логарифмический
plot_linear = True
if plot_linear == True:
    plt.plot(eps, dn_eps)
    plt.ylabel("Амплитуда")
else:
    plt.plot(eps, 20 * np.log10(dn_eps))
    plt.ylabel("Амплитуда, дБ")
plt.grid()
plt.show()

# Ширина ДН на уровне -3 дБ. Измеряется для случая свободного пространства:
# refraction = False. Чтобы получить более точный результат,
# необходимо шаг сканирования сделать чаще
if refraction == False:
    delta_eps = radar.delta_dn(eps, dn_eps)
    print(delta_eps)

# Время выполнения программы, с
toc = time.perf_counter() - tic