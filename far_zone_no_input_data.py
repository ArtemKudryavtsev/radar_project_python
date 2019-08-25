"""Зона обнаружения."""
import numpy as np
import time
import matplotlib.pyplot as plt
import radar

# Начальное время выполнения программы
tic = time.clock()

# Исходные данные

# Количество строк приемной АР
n_y_pr = 30
# Количество строк передающей АР
n_y_per = 10
# Расстояние между элементами по y, м
d_y = 0.1
# Частота, Гц
f = 1000e6
# Угол наклона антенной решётки, угл. градус
eps_a = 10
# Высота подъёма геометрического центра антенны, м
h_gc = 10
# Диаграмма направленности элемента (косинус)
dn_element = lambda angle: np.cos(angle)
# Наличие отражений от земной поверхности
refraction = True
# Глубина шероховатости, м (в дециметровом диапазоне присутствует)
h_sher = 1.5 * radar.c / f
# Лучи ЛОУ приемной АР, угл. градус
eps_beams_pr = np.arange(0.00, 31.0, 1.0, dtype=float)
# Угол, на который настроен пространственный фильтр передающей АР, угл. градус
eps_lou_per = 0.0
# Фазовое распеределение передающей АР (количество элементов фазового
# распределения равно количеству строк АР)
phase_distr_per = [0, -5.555, -3.333, 1.111, 4.444, 8.000, 12.123, 15.333, 
                    22.222, 33.333]
# Начальный угол сканирования, угл. градус
eps_start = 0.1
# Шаг сканирования, угл. градус
eps_step = 0.1
# Конечный угол сканирования, угл. градус
eps_finish = 31.0
# Массив углов места цели, угл. градус
eps = np.arange(eps_start, eps_finish + eps_step, eps_step, dtype=float)

# Параметры для расчета максимальной дальности РЛС

# Число циклов обзора в единицу времени
n_cycles = 6
# Ширина луча на уровне -3 дБ (приёмопередающая АР в азимутальной плоскости),
# угл. градус
delta_beta = 5.00
# Импульсная мощность, кВт
p_imp = 70000
# Скважность Q = T/tau
skw = 13
# Средняя мощность, Вт
p_sr = p_imp / skw
# Время облучения
t_obl = (60 / n_cycles) * (delta_beta / 360)
# ЭПР цели, м2
epr = 1
# Коэффициент усиления приёмной антенны, дБ
g_pr = 40
# Коэффициент усиления передающей антенны, дБ
g_per = 40
# Температура в градусах цельсия 
t_celsius = 20
# Шум-фактор, дБ
noise_factor = 3
# Пороговое ОСШ, дБ
snr_porog = 14
# Потери, дБ
poteri = 5

# Максимальная дальность, км
max_dist = radar.max_distance(p_sr, t_obl, g_pr, g_per, epr, f, t_celsius, 
                              noise_factor, snr_porog, poteri)

# Дальности зоны обнаружения
distances = radar.zone_distances(max_dist, n_y_pr, n_y_per, d_y, f, 
                                 eps_a, eps, eps_beams_pr, eps_lou_per, 
                                 dn_element, h_gc, refraction=refraction, 
                                 h_sher=h_sher, 
                                 phase_distr_per=phase_distr_per)

# Высоты зоны обнаружения
heights = radar.heights_of_target(eps, distances)
# Высоты, на которых находится воздушный объект
heights_const = [5, 10, 15]
# Дальности до изовысотных линий
distances_to_lines = radar.distances_to_height_const(eps, heights_const)

# Формирование изодальностных, изовысотных линий и линий постоянного угла
# и линейного массива дальностей (дальность меняется линейно от 0 до 
# максимального значения)
d_h_e_lines = True
if d_h_e_lines == True:
    # Изодальностные линии
    d_lines_x, d_lines_y = radar.dist_const_lines(eps, distances)
    # Изовысотные линии
    h_lines = radar.height_const_lines(distances, heights_const)
    # Линии постоянного угла места
    e_lines = radar.eps_const_lines(eps, distances, heights_const)
    # Линейный массив дальностей
    dists_linear = radar.distances_linear(distances)
    
# График зоны обнаружения
plt.figure()
plt.ylim([0, np.max(heights)])
plt.xlim([0, np.max(distances)])
plt.plot(distances, heights, 'b', linewidth=2)
plt.title("Зона обнаружения")
plt.xlabel("Дальность, км")
plt.ylabel("Высота, км")
# Построение изодальностных, изовысотных линий и линий постоянного угла
if d_h_e_lines == True:
    for a in np.arange(len(heights_const), dtype=int):
        plt.plot(dists_linear, h_lines[a, :], 'r')

    for a in np.arange(len(e_lines[:, 0]), dtype=int):
        plt.plot(dists_linear, e_lines[a, :], 'r--')
        
    for a in np.arange(len(d_lines_x[:, 0]), dtype=int):
        plt.plot(d_lines_x[a, :], d_lines_y[a, :],'r-.')  
plt.grid(True)  
plt.show()

# Запись в файл
with open('zone.txt', 'w') as f_out:
    for i in np.arange(len(distances), dtype=int):
        f_out.write("{:.2f}{:20.6f}".format(eps[i], distances[i]) + '\n')

# Время выполнения программы, с
toc = time.clock() - tic