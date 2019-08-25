"""Построение пеленгационной характеристики методом отношений максимальной 
амплитуды в луче к максимальной амплитуде в соседнем луче"""

import matplotlib.pyplot as plt
import numpy as np
import time
import radar

# Начальное время выполнения программы
tic = time.perf_counter()

# Количество строк АР
n_y = 30
# Расстояние между строками АР
d_y = 0.1
# Частота, Гц
f = 1000e6
# Угол наклона антенной решётки, угл. градус
eps_a = 10
# Высота подъёма геометрического центра антенны, м
h_gc = 10
# Наличие отражений от земной поверхности
refraction = True
# Глубина шероховатости, м (в дециметровом диапазоне присутствует)
h_sher = 1.5 * radar.c / f
# Количество лучей ЛОУ
n_beams = 16
# Массив углов ЛОУ (лучеобразующего устройства), угл. градус
eps_beams = np.arange(n_beams, dtype=float)
# Диаграмма направленности элемента (косинус)
dn_element = lambda angle: np.cos(angle)

# Начальный угол сканирования, угл. градус
eps_begin = 0.05
# Шаг сканирования, угл. градус
eps_step = 0.05
# Конечный угол сканирования, угл. градус
eps_finish = 10.00
# Массив углов места цели, угл. градус
eps = np.arange(eps_begin, eps_finish + eps_step, eps_step, dtype=float)

# Огибающие откликов системы пространственных фильтров
fr = radar.filters_response(n_y, d_y, f, eps_a, eps, eps_beams, dn_element,
                            h_gc, refraction=refraction, h_sher=h_sher)
    
results_path = 'peleng_neighbor_results\\'
# Считывание таблицы отношений
etalon = []
with open(results_path + 'table_plus_dots.txt', 'r') as table_file:
    for line in table_file:
        etalon.append(float(line))
delta_etalon = np.array(etalon)

# Измеренный угол места
eps_real = radar.peleng_neighbor(eps, fr, n_beams, delta_etalon)

# График пеленгационной характеристики методом отношений к соседнему максимуму
plt.figure()
plt.title("Пеленгационная характеристика (метод отношения к соседнему)")
plt.xlabel("Истинный угол места, угл. градус")
plt.ylabel("Измеренный угол места, угл. градус")
plt.plot(eps, eps_real)    
plt.grid(True)
plt.show()

# Время выполнения программы, с
toc = time.perf_counter() - tic