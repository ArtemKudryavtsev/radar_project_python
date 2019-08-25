"""Пеленгационная характеристика. Метод сканирования."""
import radar
import numpy as np
import matplotlib.pyplot as plt
import time

# Начальное время выполнения программы
tic = time.perf_counter()

# Количество строк АР
n_y = 30
# Расстояние между строками АР, м
d_y = 0.1
# Частота, Гц
f = 1000e6
# Угол наклона антенной решётки, угл. градус
eps_a = 10
# Высота подъёма геометрического центра антенны, м
h_gc = 10
# Начальный угол сканирования, угл. градус
eps_start = 0.1
# Шаг сканирования, угл. градус
eps_step = 0.1
# Конечный угол сканирования, угл. градус
eps_finish = 10.0
# Массив углов места цели, угл. градус
eps = np.arange(eps_start, eps_finish + eps_step, eps_step, dtype=float)
# Углы, на которые настроены пространственные фильтры, углю градус
eps_beams = eps
# Диаграмма направленности элемента (косинус)
dn_element = lambda angle: np.cos(angle)
# Наличие отражений от земной поверхности
refraction = True
# Глубина шероховатости, м (в дециметровом диапазоне присутствует)
h_sher = 1.5 * radar.c / f

# Истинные углы места
eps_real = radar.peleng_scan(n_y, d_y, f, eps_a, eps, eps_beams, dn_element, 
                             h_gc, refraction=refraction, h_sher=h_sher)

# График пеленгационной характеристики методом сканирования
plt.figure()
plt.title("Пеленгационная характеристика (метод сканирования)")
plt.xlabel("Истинный угол места, угл. градус")
plt.ylabel("Измеренный угол места, угл. градус")
plt.plot(eps, eps_real)
plt.grid()
plt.show()

# Время выполнения программы, с
toc = time.perf_counter() - tic