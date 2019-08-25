"""Зона обнаружения, ОСШ и потенциальные ошибки по азимуту и углу места."""
import numpy as np
import time
import matplotlib.pyplot as plt
import radar

# Начальное время выполнения программы
tic = time.perf_counter()

# Пороговое отношение сигнал/шум, дБ
snr_porog = 14
# Ширина луча приёмной антенны в угломестной плоскости на уровне -3дБ,
# угл. градус
delta_eps = 5.4
# Ширина луча приёмо-передающей антенны в азимутальной плоскости на уровне 
# -3 дБ, угл. градус
delta_beta = 5.0

# Имя файла, содержащего дальности зоны обнаружения и соответствующие 
# углы места
filename = 'zone.txt'
# Дальности зоны обнаружения и соответствующие им углы места
distances, eps = radar.zone_distances_and_angles_from_file(filename)

# Высоты зоны обнаружения
heights = radar.heights_of_target(eps, distances)
# Высоты, на которых находится воздушный объект, км
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

# Отношение сигнал/шум на дальностях до изовысотных линий
snr = radar.zone_snr(distances, distances_to_lines, snr_porog)
# График зависимости дальностей от ОСШ при фиксированных значениях высоты
plt.figure()
plt.title("Отношение сигнал/шум")
plt.xlabel("Дальность до изовысотных линий, км")
plt.ylabel("ОСШ, дБ")
for a in np.arange(len(heights_const), dtype=int):
    plt.plot(distances_to_lines[a, :], snr[a, :]);
plt.grid(True)
plt.show()

# Числовой коэффициент для расчета потенциальной ошибки по угловой координате
sigma_koefficient = np.sqrt(2 / np.pi)

# Потенциальная ошибка (по углу места)
sigma_eps = radar.zone_potencial_error(snr, delta_eps, sigma_koefficient)

# График зависимости потенциальной ошибки по углу места от дальности
plt.figure()
plt.title("Потенциальная ошибка по углу места")
plt.xlabel("Дальность до изовысотных линий, км")
plt.ylabel("Потенциальная ошибка, угл. градус")
for a in np.arange(len(heights_const), dtype=int):
    plt.plot(distances_to_lines[a, :], sigma_eps[a, :])
plt.grid(True)
plt.show()

# Потенциальная ошибка (по азимуту)
sigma_beta = radar.zone_potencial_error(snr, delta_beta, sigma_koefficient)

# График зависимости потенциальной ошибки по дальности от дальности
plt.figure()
plt.title("Потенциальная ошибка по азимуту")
plt.xlabel("Дальность до изовысотных линий, км")
plt.ylabel("Потенциальная ошибка, угл. градус")
for a in np.arange(len(heights_const), dtype=int):
    plt.plot(distances_to_lines[a, :], sigma_beta[a, :])
plt.grid(True)
plt.show()
    
# Время выполнения программы, с
toc = time.perf_counter() - tic