"""Проводка по высоте."""
import radar
import numpy as np
import matplotlib.pyplot as plt
import time

# Начальное время выполнения программы
tic = time.perf_counter()

# Исходные данные

# Количество строк АР
n_y = 30
# Расстояние между строками АР
d_y = 0.1
# Частота, Гц
freq = 1000e6
# Угол наклона антенной решётки, угл. градус
eps_a = 10
# Диаграмма направленности элемента (косинус)
dn_element = lambda angle: np.cos(angle)
# Высота подъёма геометрического центра антенны, м
h_gc = 10
# Истинная высота, км
h_0 = 10
# Начальный угол сканирования, угл. градус
eps_start = 0.1
# Шаг сканирования, угл. градус
eps_step = 0.1
# Конечный угол сканирования, угл. градус
eps_finish = 10.0
# Массив углов места цели, угл. градус
eps = np.arange(eps_start, eps_finish + eps_step, eps_step, dtype=float)
# Углы, на которые настроены пространственные фильтры , угл. градус
eps_beams = eps
# Наличие отражений от земной поверхности
refraction = True
# Глубина шероховатости, м (в дециметровом диапазоне присутствует)
h_sher = 1.5 * radar.c / freq

# Дальность до цели и высоты цели
distance, height = radar.provodka(n_y, d_y, freq, eps_a, h_0, eps, eps_beams, 
                                  dn_element, h_gc, refraction=refraction, 
                                  h_sher=h_sher)

# График проводки по высоте
plt.figure()
plt.title("Проводка по высоте")
plt.xlabel("Дальность до цели, км")
plt.ylabel("Высота полета, км")
plt.plot(distance, height)
plt.grid()
plt.show()

# Время выполнения программы, с
toc = time.perf_counter() - tic