import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd


def second_task_opt(a_1, b_1, c_1, d_1, a_2, b_2, c_2, d_2, x_0, T, n_points, r_func, eps):
    """Функция для решения задачи оптимального управления
       из второго типового расчёта для Людковского.
       Для получения решения необходимо передать условия
       из своего варианта задачи в качестве аргументов.
    Args:
        a_1 (float): коэффициент a_1 из общего вида задачи
        b_1 (float): коэффициент a_1 из общего вида задачи
        c_1 (float): коэффициент a_1 из общего вида задачи
        d_1 (float): коэффициент a_1 из общего вида задачи
        a_2 (float): коэффициент a_1 из общего вида задачи
        b_2 (float): коэффициент a_1 из общего вида задачи
        c_2 (float): коэффициент a_1 из общего вида задачи
        d_2 (float): коэффициент a_1 из общего вида задачи
        x_0 (float): начальное условие (x_0 из общего вида задачи)
        T (float): длина периода
        n_points (float): число точек для дискретизации по методу Эйлера
        r_func (function): функция для расчёта r(t) из условия задачи;
                           должная принимать в качестве аргументов как скалярные значения,
                           так и массивы numpy.array.
        eps (float): эпсилон - уровень погрешности (тоже из условия)
    Returns:
        tuple(numpy.array, numpy.array, numpy.array, numpy.array): 
            вектор времени t, вектор процесса x, вектор управления u, вектор функции фи phi
    """
    # Вектор времени
    t = np.linspace(0, T, n_points)
    # Величина шага при дискретизации
    dt = t[1] - t[0]
    
    # Вектор затрат (рассчитаем на основе переданной функции)
    r = r_func(t)
    # Вектор самого процесса x(t)
    x = np.zeros(t.shape)
    # Функция фи
    phi = np.zeros(t.shape)
    # Вектор со значениями функции управления u(t)
    u = np.zeros(t.shape)

    # Функция для оптимизации
    # Возвращает абсолютное значение отклонения от начального условия x0.
    # Поиск оптимальной траектории происходит с конца (момента времени T).
    # Нам нужно варьировать x(T) так, чтобы отклонение от x(0) от x0 было минимальным.
    def f(s):
        x[-1] = s
        
        if x[-1] >= r[-1]:
            phi[-1] = -2 * a_1 * (x[-1] - r[-1]) - c_1
        else:
            phi[-1] = -2 * b_1 * (x[-1] - r[-1]) - d_1

        for i in range(2, t.shape[0]+1):
            u_ = (phi[-i+1] - c_2) / 2 / a_2

            if u_ >= 0:
                u[-i] = u_
            else:
                u[-i] = (phi[-i+1] - d_2) / 2 / b_2

            x[-i] = x[-i+1] - dt * u[-i]
            
            if x[-i] >= r[-i]:
                phi[-i] = phi[-i+1] - dt * (2 * a_1 * (x[-i] - r[-i]) + c_1)
            else:
                phi[-i] = phi[-i+1] - dt * (2 * b_1 * (x[-i] - r[-i]) + d_1)

        delta = abs(x[0] - x_0)

        return delta

    # Находим минимум функции с заданным уровнем погрешности
    minimize(f, method='CG', x0=r[-1], tol=eps)
        
    return t, x, u, phi


if __name__ == '__main__':
    # Вариант 1
    t, x, u, phi = second_task_opt(a_1=1/2,
                                   b_1=1,
                                   c_1=1/3,
                                   d_1=7/5,
                                   a_2=2/3, 
                                   b_2=5/6, 
                                   c_2=4/3, 
                                   d_2=7/5, 
                                   x_0=1, 
                                   T=6, 
                                   n_points=100, 
                                   r_func=lambda t: 3/2 + abs(np.cos(3 * t / 2)), 
                                   eps=0.01)

    # Графики
    fig, ax = plt.subplots(3, 1)

    ax[0].plot(t, x)
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('x(t)')

    ax[1].plot(t, phi)
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('phi(t)')  

    ax[2].plot(t, u)
    ax[2].set_xlabel('t')
    ax[2].set_ylabel('u(t)')

    plt.show()

    # Результаты в виде таблицы
    df = pd.DataFrame({'t': t, 'x': x, 'u': u, 'phi': phi}) \
            .set_index('t')

    # Сохраняем в книгу Excel
    df.to_excel('second_task.xlsx', sheet_name='results')
