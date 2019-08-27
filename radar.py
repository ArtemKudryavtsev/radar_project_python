"""Модуль, описывающий характеристики радиолокационной станции.
Краткие обозначения:
АР - антенная решетка.
ДН - диаграмма направленности антенной решетки.
ЛОУ - лучеобразующее устройство."""
import numpy as np

# Коэффициент для перевода градусов в радианы
rad = np.pi/180
# Скорость света, м/с
c = 299792458
# Радиус Земли, км
radius = 6.371e3
# Эффективный радиус Земли, км
radius_eff = radius * 1.33

# Свойства поверхности, которая переотражает мешающий сигнал
# Случай для сухой земли
# Проводимость
sig = 1e-3
# Относительная диэлектрическая проницаемость
e_pr = 10

def dn_beta(n_x, d_x, freq, beta, beta_0, dn_element, weight=False):
    """Диаграмма направленности в азимутальной плоскости.
    n_x - количество столбцов антенной решетки.
    d_x - расстояние между столбцами АР, м.
    freq - частота, Гц.
    beta - массив азимутов цели, угл. градус.
    beta_0 - угол, на который установлен луч АР, угл. градус.
    dn_element - диаграмма направленности элемента АР (необходимо передать в
    качестве аргумента функцию, описывающую ДН элемента
    Пример: ДН элемента - косинус азимута цели:
    lambda angle: np.cos(angle)).
    weight - весовая обработка (по умолчанию отсутствует) (необходимо 
    передать в качестве аргумента функцию, описывающую весовую функцию).
    Пример: функция Г1.6:
    lambda n: (0.54 + 0.46 * 
    np.cos(1.6 * np.pi * (n - 0.5 * (n_x + 1)) / (n_x - 1)))."""
    # Длина волны
    lam = c / freq
    # Волновое число
    k = 2 * np.pi / lam
    # Диаграмма направленности по азимуту
    dn = np.zeros(len(beta), dtype=float)
    index = 0
    # Азимут цели
    for beta_target in beta:
        # Значение ДН на i-той итерации
        dn_i = 0
        # Проход по элементам АР
        for n in np.arange(n_x, dtype=int):
            # Диаграмма направленности элемента
            dn_elem = dn_element(beta_target * rad)
            # Весовая обработка отсутствует
            if weight == False:
                w_n = 1
            # Весовая обработка применяется
            else:
                w_n = weight(n)
            # Установка луча на азимут beta_lou
            s_0 = np.exp(-1j * k * np.sin(beta_0 * rad) * d_x * n)
            # Прямое колебание
            s_pr = np.exp(1j * k * np.sin(beta_target * rad) * d_x * n)
            # Суммарное колебание
            s = w_n * dn_elem * s_pr * s_0
            # Суммируем сигнал с выходов элементов АР
            dn_i += s
        dn[index] = np.abs(dn_i)
        index += 1
    # Возврат нормированного значения ДН
    return dn / np.max(dn)

def k_otr_coeff(eps_target, e_pr_c):
    """Вспомогательная функция для расчета коэффициентов отражения.
    eps_target - угол места цели, угл. градус.
    e_pr_c - относительная комплексная диэлектрическая проницаемость."""
    # Случай вертикальной поляризации
    k_otr_a = ((e_pr_c * np.sin(eps_target * rad) - 
                np.sqrt(e_pr_c - np.cos(eps_target * rad)**2)))
    k_otr_b = ((e_pr_c * np.sin(eps_target * rad) + 
                np.sqrt(e_pr_c - np.cos(eps_target * rad)**2)))
    return k_otr_a / k_otr_b

def k_sh_coeff(eps_target, lam, h_sher):
    """Вспомогательная функция для расчета коэффициентов шероховатости.
    eps_target - угол места цели, угл. градус.
    lam - длина волны, м.
    h_sher - глубина шероховатости."""
    return np.exp(-2 * 
                  (2 * np.pi / lam * h_sher * np.sin(eps_target * rad))**2)

def dn_eps(n_y, d_y, freq, eps_a, eps, eps_lou, dn_element, h_gc=0, 
           refraction=False, h_sher=0, phase_distr=False):
    """Диаграмма направленности в угломестной плоскости.
    n_y - количество строк антенной решетки.
    d_y - расстояние между строками АР, м.
    freq - частота, Гц.
    eps_a - угол наклона АР, угл. градус.
    eps - массив углов места цели, угл. градус.
    eps_lou - угол, на который настроен пространственный фильтр, угл. градус.
    dn_element - диаграмма направленности элемента АР.
    h_gc - высота подъема геометрического центра АР, м (h_gc связана с высотой
    поднятия нижнего края АР h_nk следующей формулой: 
    h_nk = h_gc - np.cos(eps_a * rad) * d_y * n_y / 2, для случая свободного
    пространства влияение h_gc не учитывается).
    refraction - наличие отражений от земной поверхности (по умолчанию
    отсутствует).
    h_sher - глубина шероховатости, м (шероховатость присутствует в 
    дециметровом диапазоне длин волн, в метровом диапазоне отсутствует).
    phase_distr - фазовое распределение (по умолчанию отсутствует)."""
    # Длина волны
    lam = c / freq
    # Волновое число
    k = 2 * np.pi / lam
    # Относительная комплексная диэлектрическая проницаемость
    e_pr_c = e_pr - 60j * lam * sig

    # Отклик пространственного фильтра на угол места цели (ДН)
    dn = np.zeros(len(eps), dtype=float)
    # Максимальное значение ДН в свободном пространстве
    dn_svob_max = 0
    index = 0
    # Угол цели
    for eps_target in eps:
        # Значение отклика пространственного фильтра на i-той итерации
        dn_i = 0
        # Значение отклика пространственного фильтра на i-той итерации 
        # для свободного пространства
        dn_svob_i = 0
        # Проход по элементам АР
        for n in np.arange(n_y, dtype=int):
            # Диаграмма направленности элемента
            dn_elem_pr = dn_element((eps_target - eps_a) * rad)
            # Коэффициенты пространственного фильтра w_n
            w_n = np.exp(-1j * k * np.sin((eps_lou - eps_a) * rad) * d_y * n)
            # Прямое колебание s_pr
            s_pr = np.exp(1j * k * np.sin((eps_target - eps_a) * rad) * d_y * n)
            
            # Фазовое распределение
            if phase_distr == False:
                phi_n = 1
            else:
                # Множитель фазового распределения
                a_n = 1
                # Коэффициент фазового распределения
                phi_n = a_n * np.exp(-1j * (phase_distr[n] * rad))
            
            dn_svob_i += w_n * phi_n * s_pr * dn_elem_pr
            
            # Учет отражений от земной поверхности
            if refraction == True:
                if h_sher == 0:
                    k_sh = 1
                else:
                    # Глубина шероховатости
                    h_sh = h_sher
                    # Коэффициент шероховатости k_sh
                    k_sh = k_sh_coeff(eps_target, lam, h_sh)
                # Коэффициент отражения от земной поверхности k_otr
                k_otr = k_otr_coeff(eps_target, e_pr_c)
                # Диаграмма направленности элемента мнимой АР
                dn_elem_otr = dn_element((eps_target + eps_a) * rad)
                # Отражённое колебание s_otr
                s_otr = (np.exp(-1j * k * 
                            (2 * (h_gc - np.cos(eps_a * rad) * d_y * n_y / 2) * 
                             np.sin(eps_target * rad) + 
                             np.sin((eps_target + eps_a) * rad) * d_y * n)))
                # Суммарное колебание s
                s = (w_n * phi_n * (s_pr * dn_elem_pr + 
                                    s_otr * dn_elem_otr * k_otr * k_sh))
            else:
                # Суммарное колебание s
                s = w_n * (s_pr * dn_elem_pr)
            # Суммируем сигнал с выходов элементов АР
            dn_i += s
        dn[index] = np.abs(dn_i)
        index += 1
        # Проверка на максимальное значение для ДН в свободном пространстве
        if np.abs(dn_svob_i) > dn_svob_max:
            dn_svob_max = np.abs(dn_svob_i)
    # Возврат нормированного значения ДН
    return dn / dn_svob_max

def delta_dn(angles, dn):
    """Расчет ширины ДН по уровню половинной мощности (на уровне -3 дБ).
    Чем чаще шаг сканирования, тем точнее определяется ширина ДН.
    angles - массив углов (азимут или угол места), угл. градус.
    dn - диаграмма направленности антенной решетки (азимутальная или
    угломестная плоскость)."""
    # Перевод значения ДН в логарифмический масштаб для необходимых вычислений
    dn_log = 20 * np.log10(dn)
    min_value = np.abs(dn_log[0] + 3)
    # Индекс, соответствующий значению угла на уровне -3 дБ (чем реже шаг 
    # сканированияm - выше погрешность измерения, и выбирается значение,
    # ближайшее к -3 дБ)
    index_minus_3_db = 0
    for i in np.arange(1, len(angles), dtype=int):
        if np.abs(dn_log[i] + 3) < min_value:
            min_value = np.abs(dn_log[i] + 3)
            index_minus_3_db = i
            
    return angles[index_minus_3_db] * 2

def filters_response(n_y, d_y, freq, eps_a, eps, eps_beams, dn_element, h_gc=0, 
                     refraction=False, h_sher=0):
    """Огибающие откликов пространственных фильтров.
    n_y - количество строк антенной решетки.
    d_y - расстояние между строками АР, м.
    freq - частота, Гц.
    eps_a - угол наклона АР, угл. градус.
    eps - массив углов места цели, угл. градус.
    eps_beams - углы, на которые настроены пространственные фильтры, 
    угл. градус.
    dn_element - диаграмма направленности элемента АР.
    h_gc - высота подъема геометрического центра АР, м (h_gc связана с высотой
    поднятия нижнего края АР h_nk следующей формулой: 
    h_nk = h_gc - np.cos(eps_a * rad) * d_y * n_y / 2, для случая свободного
    пространства влияение h_gc не учитывается).
    refraction - наличие отражений от земной поверхности (по умолчанию
    отсутствует).
    h_sher - глубина шероховатости, м (шероховатость присутствует в 
    дециметровом диапазоне длин волн, в метровом диапазоне отсутствует)."""
    # Длина волны
    lam = c / freq
    # Волновое число
    k = 2 * np.pi / lam
    # Относительная комплексная диэлектрическая проницаемость
    e_pr_c = e_pr - 60j * lam * sig
    
    # Огибающие откликов пространственных фильтров
    fr = np.zeros((len(eps_beams), len(eps)), dtype=float)
    index_target = 0
    # Угол цели
    for eps_target in eps:
        # Отклик пространственного фильтра на угол места цели (ДН)
        fr_target = np.zeros(len(eps_beams), dtype=float)
        index_lou = 0
        for eps_lou in eps_beams:
            # Значение отклика пространственного фильтра на i-той итерации
            y_i = 0
            # Проход по элементам АР
            for n in np.arange(n_y, dtype=int):
                # Диаграмма направленности элемента
                dn_elem_pr = dn_element((eps_target - eps_a) * rad)
                # Коэффициенты пространственного фильтра w_n
                w_n = (1 / n_y * 
                       np.exp(-1j * k * 
                              np.sin((eps_lou - eps_a) * rad) * d_y * n))
                # Прямое колебание s_pr
                s_pr = np.exp(1j * k * 
                              np.sin((eps_target - eps_a) * rad) * d_y * n)
                # Учет отражений от земной поверхности
                if refraction == True:
                    if h_sher == 0:
                        k_sh = 1
                    else:
                        # Глубина шероховатости
                        h_sh = h_sher
                        # Коэффициент шероховатости k_sh
                        k_sh = k_sh_coeff(eps_target, lam, h_sh)
                    # Коэффициент отражения от земной поверхности k_otr
                    k_otr = k_otr_coeff(eps_target, e_pr_c)
                    # Диаграмма направленности элемента мнимой АР
                    dn_elem_otr = dn_element((eps_target + eps_a) * rad)
                    # Отражённое колебание s_otr
                    s_otr = (np.exp(-1j * k * 
                            (2 * (h_gc - np.cos(eps_a * rad) * d_y * n_y / 2) * 
                             np.sin(eps_target * rad) + 
                             np.sin((eps_target + eps_a) * rad) * d_y * n)))
                    # Суммарное колебание s
                    s = (w_n * (s_pr * dn_elem_pr + 
                                s_otr * dn_elem_otr * k_otr * k_sh))
                else:
                    # Суммарное колебание s
                    s = w_n * (s_pr * dn_elem_pr)
                # Суммируем сигнал с выходов элементов АР
                y_i = y_i + s
            fr_target[index_lou] = np.abs(y_i)
            index_lou += 1
        fr[:, index_target] = fr_target
        index_target += 1
    return fr

def peleng_scan(n_y, d_y, freq, eps_a, eps, eps_beams, dn_element, h_gc = 0, 
                refraction=False, h_sher=0):
    """Пеленгационная характеристика методом сканирования.
    n_y - количество строк антенной решетки.
    d_y - расстояние между строками АР, м.
    freq - частота, Гц.
    eps_a - угол наклона АР, угл. градус.
    eps - массив углов места цели, угл. градус.
    eps_beams - углы, на которые настроены пространственные фильтры, 
    угл. градус.
    dn_element - диаграмма направлености элемента АР.
    h_gc - высота подъема геометрического центра АР, м (h_gc связана с высотой
    поднятия нижнего края АР h_nk следующей формулой: 
    h_nk = h_gc - np.cos(eps_a * rad) * d_y * n_y / 2, для случая свободного
    пространства влияение h_gc не учитывается).
    refraction - наличие отражений от земной поверхности (по умолчанию
    отсутствует).
    h_sher - глубина шероховатости, м (шероховатость присутствует в 
    дециметровом диапазоне длин волн, в метровом диапазоне отсутствует)."""
    # Шаг сканирования
    eps_step = eps[1] - eps[0]
    # Массив измеренных углов места
    eps_real = np.zeros(len(eps), dtype=float)
    
    # Огибающие откликов пространственных фильтров
    fr = filters_response(n_y, d_y, freq, eps_a, eps, eps_beams, dn_element, 
                               h_gc, refraction, h_sher)
    
    for i in np.arange(len(eps), dtype=int):
        # Индекс, соответствующий максимальной амплитуде в массиве
        n_max = fr[:,i].argmax()
        eps_real[i] = eps_step * (n_max + 1)
    return eps_real

def ratio_table_pairs(eps, fr, n_beams):
    """Формирование таблицы отношений методом отношений максимальных амплитуд 
    в паре лучей.
    eps - массив углов места цели, угл. градус.
    fr - огибающие откликов системы пространственных фильтров в 
    заданном диапазоне углов места цели.
    n_beams - количество лучей ЛОУ."""
    # Начальный угол сканирования, угл. градус
    eps_start = eps[0]
    # Шаг сканирования, угл. градус
    eps_step = eps[1] - eps[0]
    # Конечный угол сканирования, угл .градус
    eps_finish = eps[-1]
    
    # Массив отношений амплитуд
    delta_etalon = np.zeros(len(eps), dtype=float)
    # Массив углов места, угл. градус
    angles = np.zeros(len(eps), dtype=float)
    # Индекс таблицы отношений
    index_table = 0
    
    # Массив отношений амплитуд с точками пересечений
    delta_etalon_with_dots = []
    # Точки пересечения
    dots = []
    
    # Номер конца массива
    max_eps = int((eps_finish - eps_start) * (1 / eps_step) + 1)
    # Первый элемент массива
    min_eps = 0
    # Номер луча с максимальной амплитудой
    beam = 0
    for i in np.arange(min_eps, max_eps, dtype=int):
        # Текущий угол
        the_angle = i * eps_step + eps_start
        # Номер пространственного фильтра
        for j in np.arange(n_beams, dtype=int):
            # Если максимум в j-м пространственном фильтре
            if fr[j, i] == np.max(fr[:, i]):
                # Если не точка пересечения, и максимум в первом луче k-й пары
                if (beam == j) and (beam % 2 == 0):
                    # Амплитуда в первом фильтре k-й пары
                    a_1 = fr[j, i]
                    # Амплитуда во втором фильтре k-й пары
                    a_2 = fr[j + 1, i]
                # Если не точка пересечения, и максимум во втором луче k-й пары
                elif (beam == j) and (beam % 2 == 1):
                    # Амплитуда в первом фильтре k-й пары
                    a_1 = fr[j - 1, i]
                    # Амплитуда во втором фильтре k-й пары
                    a_2 = fr[j, i]
                # Если точка пересечения
                elif (beam != j):
                    # Максимум был в первом луче k-й пары
                    if beam % 2 == 0:
                        # Амплитуда в первом фильтре k-й пары
                        a_1 = fr[j - 1, i]
                        # Амплитуда во втором фильтре k-й пары
                        a_2 = fr[j, i]
                    # Максимум был во втором луче k-й пары
                    elif beam % 2 == 1:
                        # Амплитуда в первом фильтре k-й пары
                        a_1 = fr[j, i]
                        # Амплитуда во втором фильтре k-й пары
                        a_2 = fr[j + 1, i]
                        # Добавляем точку пересечения в массив
                        delta_etalon_with_dots.append(0.0)
                    # Точки пересечения
                    dots.append(the_angle)
                    # Устанавливаем номер луча с максимальной амплитудой
                    beam = j
                # Отношение амплитуд
                result = a_1 / a_2
                delta_etalon_with_dots.append(result)
                # Массив отношений амплитуд
                delta_etalon[index_table] = result
                # Массив углов места
                angles[index_table] = the_angle
                break
        index_table += 1
    return (delta_etalon, angles, dots, delta_etalon_with_dots)

def peleng_pairs(eps, fr, n_beams, delta_etalon):
    """Построение пеленгационной характеристики методом отношений максимальных 
    амплитуд в паре лучей.
    eps - массив углов места цели, угл. градус.
    fr - огибающие откликов системы пространственных фильтров в 
    заданном диапазоне углов места цели.
    n_beams - количество лучей ЛОУ.
    delta_etalon - таблица отношений."""
    # Начальный угол сканирования, угл. градус
    eps_start = eps[0]
    # Шаг сканирования, угл. градус
    eps_step = eps[1] - eps[0]
    # Конечный угол сканирования, угл .градус
    eps_finish = eps[-1]
    
    # Номер конца массива
    max_eps = int((eps_finish - eps_start) * (1 / eps_step) + 1)
    # Первый элемент массива    
    min_eps = 0
    # Массив реальных углов места, угл. градус
    eps_real = np.zeros(len(eps), dtype=float)
    # Расчёт и построение пеленгационной характеристики
    for i in np.arange(min_eps, max_eps, dtype=int):
        # Номер пространственного фильтра
        for j in np.arange(n_beams, dtype=int):
            result = 0
            # Если максимум в j-м пространственном фильтре
            if fr[j, i] == np.max(fr[:, i]):
                # Если луч первый в паре
                if j % 2 == 0:
                    # Начало отрезка таблицы, соответствующий выбранной паре
                    # лучей
                    index = j / 2
                    # Амплитуда в первом фильтре k-й пары
                    a_1 = fr[j, i]
                    # Амплитуда во втором фильтре k-й пары  
                    a_2 = fr[j + 1, i]
                # Если луч второй в паре
                else:
                    # Начало отрезка таблицы, соответствующий выбранной паре
                    # лучей
                    index = (j + 1) / 2 - 1
                    # Амплитуда в первом фильтре k-й пары
                    a_1 = fr[j - 1, i]
                    # Амплитуда во втором фильтре k-й пары
                    a_2 = fr[j, i]
                # Отношение амплитуд
                result = a_1 / a_2
                
                # Количество знаков после запятой
                z = 6
                # Задание количества знаков после запятой
                result = np.round(result, decimals=z)
                # Минимальная разница (задаем очень большое 
                # значение с запасом, чтобы разница delta_etalon[k] - result
                # была гарантированно меньше, чем minimal на первом шаге),
                # чтобы установить минимальную разницу для выбранного отрезка,
                # равной delta_etalon[k] - result для первого значения из
                # отрезка
                minimal = 1e9
                # Реальное значение угла места
                eps_real_i = 0
                # Количество встреченных точек пересечения
                dot_count = 0
                    
                # Выполняется проход по значениям таблицы
                for k in np.arange(len(delta_etalon), dtype=int):
                    # Текущий угол места
                    the_angle = (k - dot_count) * eps_step + eps_start
                    # Если точка пересечения (в данном случае 0)
                    if delta_etalon[k] == 0:
                        # Увеличить количество точек пересечения
                        dot_count += 1
                    # Если не точка пересечения и начало необходимого отрезка
                    elif (delta_etalon[k] != 0) and (dot_count == index):
                        # Проверка на минимальную разность между значением из 
                        # таблицы и рассчитанным отношением
                        if np.abs(delta_etalon[k] - result) <= minimal:
                            # Новое минимальное значение
                            minimal = np.abs(delta_etalon[k] - result)
                            # Угол места, соответствующий минимальной разности
                            eps_real_i = the_angle
                    # Если выход за пределы отрезка - выход из цикла
                    elif dot_count == (index + 1):
                        break
                eps_real[i] = eps_real_i
                break
    return eps_real

def ratio_table_neighbor(eps, fr, n_beams):
    """Формирование таблицы отношений методом отношений максимальной амплитуды 
    в луче к максимальной амплитуде в соседнем луче.
    eps - массив углов места цели, угл. градус.
    fr - огибающие откликов системы пространственных фильтров в 
    заданном диапазоне углов места цели.
    n_beams - количество лучей ЛОУ."""
    # Начальный угол сканирования, угл. градус
    eps_start = eps[0]
    # Шаг сканирования, угл. градус
    eps_step = eps[1] - eps[0]
    # Конечный угол сканирования, угл .градус
    eps_finish = eps[-1]
    
    # Массив отношений амплитуд
    delta_etalon = np.zeros(len(eps), dtype=float)
    # Массив углов места, угл. градус
    angles = np.zeros(len(eps), dtype=float)
    # Индекс таблицы отношений
    index_table = 0
    
    # Массив отношений амплитуд с точками пересечений
    delta_etalon_with_dots = []
    # Точки пересечения
    dots = []
    
    # Номер конца массива
    max_eps = int((eps_finish - eps_start) * (1 / eps_step) + 1)
    # Первый элемент массива
    min_eps = 0
    # Флаг, необходимый для перехода от вычислений между одной парой лучей к
    # другой
    flag = False
    for i in np.arange(min_eps, max_eps, dtype=int):
        # Текущий угол
        the_angle = i * eps_step + eps_start
        # Номер пространственного фильтра
        for j in np.arange(n_beams, dtype=int):
            # Если максимум в j-м пространственном фильтре
            if fr[j, i] == np.max(fr[:, i]):
                # Если максимум в первом луче
                if j == 0:
                    index = j
                    # Амплитуда в первом луче
                    a_1 = fr[j, i]
                    # Амплитуда во втором луче
                    a_2 = fr[j + 1, i]
                    # Отношение амплитуд
                    result = a_1 / a_2
                # Если максимум не в первом луче
                else:
                    if index == 0:
                        dots.append(the_angle);
                        delta_etalon_with_dots.append(0.0)
                    index = j
                    # Если не последний луч
                    if j != n_beams - 1:
                        # Амлитуда в j-м луче (максимум)
                        a_1 = fr[j, i]
                        # Амплитуда в (j-1)-м луче
                        a_2 = fr[j - 1, i]
                        # Амплитуда в (j+1)-м луче
                        a_3 = fr[j + 1, i]
                        # Если левый соседний от центрального луч больше 
                        # правого соседнего и не было перехода через точку
                        # пересечения лучей
                        if (a_2 >= a_3) and (flag == False):
                            result = a_1 / a_2
                        # Если левый соседний от центрального луч больше 
                        # правого соседнего и был переход через точку
                        # пересечения лучей
                        elif (a_2 >= a_3) and (flag == True):
                            flag = False
                            dots.append(the_angle)
                            delta_etalon_with_dots.append(0.0)
                            result = a_1 / a_2
                        # Если левый соседний от центрального луч меньше 
                        # правого соседнего и был переход через точку
                        # пересечения лучей
                        elif (a_2 < a_3) and (flag == False):
                            flag = True
                            dots.append(the_angle)
                            delta_etalon_with_dots.append(0);
                            result = a_1 / a_3
                        # Если левый соседний от центрального луч меньше 
                        # правого соседнего и не было перехода через точку
                        # пересечения лучей
                        elif (a_2 < a_3) and (flag == True):
                            result = a_1 / a_3
                    # Если максимум в последнем луче
                    else:
                        if flag == True:
                            flag = False
                            dots.append(the_angle)
                            delta_etalon_with_dots.append(0.0)
                        a_1 = fr[j,i]
                        a_2 = fr[j-1,i]
                        result = a_1 / a_2      
                delta_etalon_with_dots.append(result)
                delta_etalon[index_table] = result
                angles[index_table] = the_angle
        index_table += 1
    return (delta_etalon, angles, dots, delta_etalon_with_dots)

def peleng_neighbor(eps, fr, n_beams, delta_etalon):
    """Построение пеленгационной характеристики методом отношений максимальной 
    амплитуды в луче к максимальной амплитуде в соседнем луче.
    eps - массив углов места цели, угл. градус.
    fr - огибающие откликов системы пространственных фильтров в 
    заданном диапазоне углов места цели.
    n_beams - количество лучей ЛОУ.
    delta_etalon - таблица отношений."""
    # Начальный угол сканирования, угл. градус
    eps_start = eps[0]
    # Шаг сканирования, угл. градус
    eps_step = eps[1] - eps[0]
    # Конечный угол сканирования, угл .градус
    eps_finish = eps[-1]
    
    # Номер конца массива
    max_eps = int((eps_finish - eps_start) * (1 / eps_step) + 1)
    # Первый элемент массива
    min_eps = 0
    # Массив реальных углов места, угл. градус
    eps_real = np.zeros(len(eps), dtype=float)  
    # Расчёт и построение пеленгационной характеристики
    for i in np.arange(min_eps, max_eps, dtype=int):
        # Номер пространственного фильтра
        for j in np.arange(n_beams, dtype=int):
            # Если максимум в j-м пространственном фильтре
            if fr[j, i] == np.max(fr[:, i]):
                # Максимум в первом луче
                if j == 0:
                    # Амплитуда в первом луче
                    a_1 = fr[j, i]
                    # Амплитуда во втором луче
                    a_2 = fr[j + 1, i]
                    # Отношение амплитуд
                    result = a_1 / a_2
                    # Номер точки пересечения, соответствующей концу
                    # первого отрезка из таблицы
                    index = 1
                # Если максимум в последнем луче
                elif j == n_beams - 1:
                    # Амплитуда в последнем луче
                    a_1 = fr[j, i]
                    # Амплитуда в предпоследнем луче
                    a_2 = fr[j - 1, i]
                    # Отношение амплитуд
                    result = a_1 / a_2
                    # Номер точки пересечения, соответствующей концу
                    # последнего отрезка из таблицы
                    index = n_beams * 2 - 2
                # Не первый и не последний лучи
                else:
                    # Первый максимум (j-й элемент)
                    a_1 = fr[j, i]
                    # Значение для j-1 элемента
                    a_2 = fr[j - 1, i]
                    # Значение для j+1 элемента
                    a_3 = fr[j + 1, i]
                    # Если значение j-1 элемента больше значения j+1 элемента
                    if a_2 >= a_3:
                        # Отношение амплитуд
                        result = a_1 / a_2
                        # Номер точки пересечения, соответствующей концу
                        # необходимого отрезка из таблицы
                        index = (j + 1) * 2 - 2
                    elif a_2 < a_3:
                        # Отношение максимумов
                        result = a_1 / a_3
                        # Номер точки пересечения, соответствующей концу
                        # необходимого отрезка из таблицы
                        index = (j + 1) * 2 - 1
                # Счетчик точек пересечения
                dot_count = 0
                # Количество знаков после запятой
                z = 6
                # Задание количества знаков после запятой
                result = np.round(result, decimals=z)
                # Минимальная разница (задаем очень большое 
                # значение с запасом, чтобы разница delta_etalon[k] - result
                # была гарантированно меньше, чем minimal на первом шаге),
                # чтобы установить минимальную разницу для выбранного отрезка,
                # равной delta_etalon[k] - result для первого значения из
                # отрезка
                minimum = 1e9
                # Выполняется проход по значениям таблицы
                for k in np.arange(len(delta_etalon), dtype=int):
                    # Пока не достигли начала нужного отрезка, увеличиваем
                    # количество пройденных точек
                    if (delta_etalon[k] == 0) and (dot_count != index):
                        dot_count += 1 #Увеличиваем счётчик единиц
                    # Если значение таблицы не точка пересечения и не конец 
                    # массива и мы находимся в нужном отрезке
                    elif ((delta_etalon[k] != 0) and 
                          (dot_count == index - 1) and 
                          (k != len(delta_etalon) - 1)):
                        if np.abs(result - delta_etalon[k]) < minimum:
                            # Минимальная разность
                            minimum = np.abs(result - delta_etalon[k])
                            # Текущий угол места
                            the_angle = (k - dot_count) * eps_step + eps_start
                    # Если конец соответствующего отрезка таблицы отношений,
                    # либо если достигли конца таблицы
                    elif ((dot_count == index) or (((delta_etalon[k] != 0) and 
                           (dot_count == index - 1) 
                           and (k == len(delta_etalon) - 1)))):
                        # Последняя проверка на минимальную разность для
                        # последнего элемента таблицы
                        if np.abs(result - delta_etalon[k]) < minimum:
                            # Минимальная разность
                            minimum = np.abs(result - delta_etalon[k])
                            # Рассчитанный угол места
                            the_angle = (k - dot_count) * eps_step + eps_start
                        # Запись рассчитанного угла места в массив
                        eps_real[i] = the_angle
                        break
                break
    return eps_real

def heights_of_target(eps, distances, curvature=False):
    """Высота полета цели.
    eps - угол места цели, угл. градус.
    distances - дальность до цели, км.
    curvature - учет кривизны земной поверхности."""
    heights = np.zeros(len(eps), dtype=float)
    for i in np.arange(len(heights), dtype=int):
        if curvature == True:
            heights[i] = (distances[i] * np.sin(eps[i] * rad) + 
                  distances[i] ** 2 / (2 * radius_eff))
        else:
            heights[i] = distances[i] * np.sin(eps[i] * rad)    
    return heights

def provodka(n_y, d_y, freq, eps_a, h_0, eps, eps_beams, dn_element, h_gc=0, 
             refraction=False, h_sher=0):
    """Пеленгационная характеристика методом сканирования.
    n_y - количество строк антенной решетки.
    d_y - расстояние между строками АР, м.
    freq - частота, Гц.
    eps_a - угол наклона АР, угл. градус.
    h_0 - высота полета цели, км.
    eps - массив углов места цели, угл. градус.
    eps_beams - углы, на которые настроены пространственный фильтр, угл. 
    градус.
    dn_element - диаграмма направленности элемента АР.
    h_gc - высота подъема геометрического центра АР, м (h_gc связана с высотой
    поднятия нижнего края АР h_nk следующей формулой: 
    h_nk = h_gc - np.cos(eps_a * rad) * d_y * n_y / 2, для случая свободного
    пространства влияение h_gc не учитывается).
    refraction - наличие отражений от земной поверхности (по умолчанию
    отсутствует).
    h_sher - глубина шероховатости, м (шероховатость присутствует в 
    дециметровом диапазоне длин волн, в метровом диапазоне отсутствует)."""
    # Измеренный угол места цели
    eps_real = peleng_scan(n_y, d_y, freq, eps_a, eps, eps_beams, dn_element, 
                           h_gc, refraction, h_sher)
    
    # Дальности до теоретической цели на высоте h_0
    distances = np.zeros(len(eps), dtype=float)
    for i in np.arange(len(distances), dtype=int):
        distances[i] = (radius_eff * (-np.sin(eps[i] * rad) + 
                np.sqrt((np.sin(eps[i] * rad) ** 2) + 2 * h_0 / radius_eff)))
    
    # Действительная измеренная высота цели
    height = heights_of_target(eps_real, distances, curvature=True)
        
    return distances, height

def max_distance(p_sr, t_obl, g_pr, g_per, epr, freq, t_celsius, noise_factor, 
                 snr_porog, poteri):
    """Расчет максимальной дальности обнаружения.
    p_sr - средняя мощность, Вт.
    t_obl - время облучения.
    g_pr - коэффициент усиления приёмной антенны, дБ.
    g_per - коэффициент усиления передающей антенны, дБ.
    epr - ЭПР цели, м2.
    freq - частота, Гц.
    t_celsius - температура в градусах цельсия .
    noise_factor - шум-фактор, дБ.
    snr_porog - пороговое ОСШ, дБ.
    poteri - потери, дБ."""
    # Длина волны
    lam = c / freq
    # Постоянная Больцмана
    k_bolc = 1.38e-23
    # Температура в кельвинах
    t_kelvin = t_celsius + 273.15
    
    # Максимальная дальность в км
    return np.sqrt(np.sqrt(((p_sr * t_obl * 10 ** (g_pr / 10) * 
                             10 ** (g_per / 10) * epr * lam ** 2) / 
                      (((4 * np.pi) ** 3) * 10 ** (noise_factor / 10) * 
                       k_bolc * t_kelvin * 10 ** (snr_porog / 10) * 
                       10 ** (poteri / 10))))) / 1000

def zone_distances_and_angles_from_file(filename):
    """Считывание значений дальностей зоны обнаружения и соответствующих им
    значений углов места из файла.
    Файл должен иметь два столбца, первый из которых - угол места, второй
    столбец - дальность.
    filename - имя файла."""
    with open(filename, 'r') as freq:
        distances_angles = list()
        for line in freq:
            a, b = line.split()
            distances_angles.append(float(a))
            distances_angles.append(float(b))
    
    # Массив дальностей          
    distances = np.zeros(len(distances_angles) // 2, dtype=float)
    # Массив углов
    angles = np.zeros(len(distances_angles) // 2, dtype=float)
    index_angle = 0
    index_dance = 0
    for i in np.arange(len(distances_angles)):
        if i % 2 == 0:
            angles[index_angle] = distances_angles[i]
            index_angle += 1
        else:
            distances[index_dance] = distances_angles[i]
            index_dance += 1
    
    # Начальный угол сканирования, угл. градус
    eps_start = angles[0]
    # Шаг сканирования, угл. градус
    eps_step = angles[1] - angles[0]
    # Конечный угол сканирования, угл. градус
    eps_finish = angles[len(angles)-1]
    # Массив углов места цели, угл. градус
    eps = np.arange(eps_start, eps_finish + eps_step, eps_step, dtype=float)
    
    return distances, eps
          
def zone_distances(max_dist, n_y_pr, n_y_per, d_y, freq, eps_a, eps, 
                   eps_beams_pr, eps_lou_per, dn_element, h_gc=0, 
                   refraction=False, h_sher=0, phase_distr_per=False):
    """Расчет дальностей зоны обнаружения.
    max_dist - максимальная дальность, км.
    n_y_pr - количество строк приемной АР.
    n_y_per - количество строк передающей АР.
    d_y - расстояние между элементами АР, м.
    freq - частота, Гц.
    eps_a - угол наклона АР, угл. градус.
    eps - угол места цели, угл. градус.
    eps_beams_pr - углы ЛОУ приемной АР, угл. градус.
    eps_lou_per - угол, на который настроен пространственный фильтр
    передающей АР, угл. градус.
    dn_element - диаграмма направленности элемента АР.
    h_gc - высота подъема геометрического центра АР, м (h_gc связана с высотой
    поднятия нижнего края АР h_nk следующей формулой: 
    h_nk = h_gc - np.cos(eps_a * rad) * d_y * n_y / 2, для случая свободного
    пространства влияение h_gc не учитывается).
    refraction - наличие отражений от земной поверхности (по умолчанию
    отсутствует).
    h_sher - глубина шероховатости, м (шероховатость присутствует в 
    дециметровом диапазоне длин волн, в метровом диапазоне отсутствует).
    phase_distr_per - наличие фазового распределения передающей АР (по 
    умолчанию отсутствует)."""
    # Огибающие откликов системы пространственных фильтров для приемной АР
    fr_eps_pr = filters_response(n_y_pr, d_y, freq, eps_a, eps, eps_beams_pr, 
                                    dn_element, h_gc, refraction, h_sher)
    # Приемная ДН (огибающая максимумов откликов системы пространственных
    # фильтров приемной АР)
    dn_eps_pr = np.zeros(len(eps), dtype=float)
    for i in np.arange(len(eps), dtype=int):
        dn_eps_pr[i] = np.max(fr_eps_pr[:,i])
      
    # Передающая ДН
    dn_eps_per = dn_eps(n_y_per, d_y, freq, eps_a, eps, eps_lou_per, dn_element, 
                        h_gc, refraction, h_sher, phase_distr_per)
    
    # Приемо-передающая ДН
    dn_pr_per = np.zeros(len(eps), dtype=float)
    for i in np.arange(len(eps), dtype=int):
        dn_pr_per[i] = np.sqrt(dn_eps_pr[i] * dn_eps_per[i])
    
    # Дальности зоны обнаружения
    distances = np.zeros(len(eps), dtype=float)
    for i in np.arange(len(eps), dtype=int):
        distances[i] = max_dist * dn_pr_per[i]     
    return distances

def distances_to_height_const(eps, heights_const):
    """Формирование дальностей до изовысотных линий.
    eps - угол места цели, угл. градус.
    heights_const - высоты полета цели, км."""
    # Дальности до изовысотных линий
    dist_to_h = np.zeros((len(heights_const), len(eps)))
    index = 0
    for h_0_i in heights_const:
        dist_to_h_i = np.zeros(len(eps))
        index_i = 0
        for eps_i in eps:
            dist_to_h_i[index_i] = (radius_eff * (-np.sin(eps_i * rad) + 
                                    np.sqrt((np.sin(eps_i * rad) ** 2) + 
                                            2 * h_0_i / radius_eff)))
            index_i += 1
        dist_to_h[index] = dist_to_h_i
        index += 1
    return dist_to_h

def distances_linear(distances):
    """Массив дальностей, значения которого линейно изменяются от минимальной 
    до максимально возможной дальности.
    Необходим для построения графиков изовысотных линий и линий постоянного 
    угла места."""
    dist_begin = 0
    dist_step = np.max(distances) / len(distances)
    dist_end = np.max(distances)
    return np.arange(dist_begin, dist_end, dist_step)

def dist_const_lines(eps, distances, dist_const_step=100):
    """Формирование изодальностных линий.
    eps - угол места цели, угл. градус.
    distances - дальность до цели, км.
    dist_const_step - шаг формирования изодальностных линий, км."""
    dist_const_begin = 0
    dist_const_end = np.max(distances)
    
    # Дальности, через которые будут проходить изодальностные линии
    distance_const = np.arange(dist_const_begin, dist_const_end, 
                               dist_const_step)
    #Изодальностные линии
    d_lines_x = np.zeros((len(distance_const), len(distances)))
    d_lines_y = np.zeros((len(distance_const), len(distances)))
    index_d = 0
    for dist_const_i in distance_const:
        d_lines_x_i = np.zeros(len(distances))
        d_lines_y_i = np.zeros(len(distances))
        index_d_i = 0
        for eps_i in eps:
            d_lines_x_i[index_d_i] = dist_const_i * np.cos(eps_i*rad)
            d_lines_y_i[index_d_i] = dist_const_i * np.sin(eps_i*rad)
            index_d_i += 1
        d_lines_x[index_d] = d_lines_x_i
        d_lines_y[index_d] = d_lines_y_i
        index_d += 1    
    return d_lines_x, d_lines_y

def height_const_lines(distances, heights_const):
    """Формирование изовысотных линий.
    distances - дальность до цели, км.
    heights_const - высоты полета цели, км."""
    # Массив дальностей от минимальной до максимальной
    dists = distances_linear(distances)
    
    # Изовысотные линии
    h_lines = np.zeros((len(heights_const), len(distances)))
    index_h = 0
    for h_0 in heights_const:
        h_lines_h_0 = np.zeros(len(distances))
        index_h_i = 0
        for i in dists:
            h_lines_i = h_0 - (i ** 2) / (2 * radius_eff)
            h_lines_h_0[index_h_i] = h_lines_i
            index_h_i += 1
        h_lines[index_h] = h_lines_h_0
        index_h += 1
    return h_lines
    
def eps_const_lines(eps, distances, heights_const, eps_const_step=1.0):
    """Формирование линий постоянного угла места.
    eps - угол места цели, угл. градус.
    distances - дальность до цели, км.
    heights_const - высоты полета цели, км.
    eps_const_step - шаг формирования линий постоянного угла места, 
    угл. градус."""
    # Начальный угол сканирования, угл. градус
    eps_const_start = eps[0]
    # Конечный угол сканирования, угл. градус
    eps_const_finish = eps[-1]
    epsilon = np.arange(eps_const_start, eps_const_finish, eps_const_step)
    
    # Массив дальностей от минимальной до максимальной
    dists = distances_linear(distances)
    
    # Линии постоянного угла места
    eps_lines = np.zeros((len(epsilon), len(distances)))
    index_eps = 0
    for eps_const in epsilon:
        eps_lines_h_0 = np.zeros(len(distances))
        index_eps_i = 0
        for i in dists:
            eps_lines_i = i * np.sin(eps_const * rad)
            eps_lines_h_0[index_eps_i] = eps_lines_i
            index_eps_i += 1
        eps_lines[index_eps] = eps_lines_h_0
        index_eps += 1
    return eps_lines

def zone_snr(distances, distances_to_lines, snr_porog):
    """Отношение сигнал/шум на дальностях до изовысотных линий.
    distances - дальности зоны обнаружения, км.
    distances_to_lines - дальности до изовысотных линий, км.
    snr_porog - пороговое ОСШ на границе зоны обнаружения, дБ."""
    # Размерность, соответствующая количеству изовысотных линий
    h_size = len(distances_to_lines[:, 0])
    # Размерность, соответствующая количеству дальностей в массиве дальностей 
    d_size = len(distances_to_lines[0, :])
    
    # Отношение сигнал/шум, дБ
    snr = np.zeros((h_size, d_size), dtype=float)
    for i in np.arange(h_size, dtype=int):
        # ОСШ на выбранной высоте h_0
        snr_h_0 = np.zeros(d_size, dtype=float)
        for j in np.arange(d_size, dtype=int):
            snr_j = (10 * np.log10((distances[j] ** 4) * 
                                   (10 ** (snr_porog / 10)) /
                                   (distances_to_lines[i, j] ** 4)))
            # Если расчетное ОСШ больше или равно пороговому
            if snr_j >= snr_porog:
                snr_h_0[j] = snr_j
            # Если выход за границу зоны обнаружения
            else:
                snr_h_0[j] = np.NaN
        snr[i] = snr_h_0
    return snr

def zone_potencial_error(snr, delta_dn, sigma_coeff):
    """Потенциальная ошибка измерения угловой координаты.
    snr - отношение сигнал/шум на дальностях до изовысотных линий, дБ.
    delta_dn - ширина ДН на уровне -3 дБ, угл. градус.
    sigma_coeff - числовой коэффициент."""
    # Размерность, соответствующая количеству изовысотных линий
    h_size = len(snr[:, 0])
    # Размерность, соответствующая количеству дальностей в массиве дальностей 
    d_size = len(snr[0, :])
    # Потенциальная ошибка, угл. градус
    sigma = np.zeros((h_size, d_size), dtype=float)
    for i in np.arange(h_size, dtype=int):
        # Потенциальная ошбка на выбранной высоте h_0
        sigma_h_0 = np.zeros(d_size, dtype=float)
        for j in np.arange(d_size, dtype=int):
            sigma_h_0[j] = (sigma_coeff * delta_dn / 
                        (10 ** (snr[i, j] / 20)))
        sigma[i] = sigma_h_0
    return sigma