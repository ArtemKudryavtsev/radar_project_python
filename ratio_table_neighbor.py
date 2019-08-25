"""Формирование таблицы отношений методом отношений 
максимальной амплитуды в луче к максимальной амплитуде в соседнем луче."""

import matplotlib.pyplot as plt
import numpy as np
import time
import radar

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
fr = radar.filters_response(n_y, d_y, f, eps_a, eps, eps_beams, dn_element)

# График огибающих откликов пространственных фильтров
dn_plot = True
if dn_plot == True:
    # График
    plt.figure()
    plt.title("Огибающие откликов пространственных фильтров")
    plt.xlabel("Угол места, угл. градус")
    plt.ylabel("Амплитуда")
    for i in np.arange(n_beams, dtype=int):
        plt.plot(eps, fr[i, :])
    plt.grid(True)
    plt.show()

# Расчет таблицы отношений
# delta_etalon - массив отношений
# angles - массив углов места
# dots - точки пересечения
# delta_etalon_with_dots - массив отношений с точками пересечения
delta_etalon, angles, dots, delta_etalon_with_dots = radar.ratio_table_neighbor(eps, fr, n_beams)

# Запись в файл
results_path = 'peleng_neighbor_results\\'
# Таблица
with open(results_path + 'table.txt', 'w') as table_file:
    for i in np.arange(len(delta_etalon), dtype=int):
        table_file.write("{:.6f}{:6.2f}".format(delta_etalon[i], angles[i]) + '\n')

# Точки пересечения
with open(results_path + 'dots.txt', 'w') as dots_file:
    for i in np.arange(len(dots), dtype=int):
        dots_file.write("{:5.2f}".format(dots[i]) + '\n')

# Таблица отношений с точками пересечений в виде единиц
with open(results_path + 'table_plus_dots.txt', 'w') as table_plus_dots_file:
    for i in np.arange(len(delta_etalon_with_dots), dtype=int):
        table_plus_dots_file.write("{:.6f}".format(delta_etalon_with_dots[i]) + '\n')

# Время выполнения программы, с
toc = time.perf_counter() - tic