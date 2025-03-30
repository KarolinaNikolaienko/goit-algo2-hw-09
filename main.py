import random
import math

import numpy as np


# Визначення функції Сфери
def sphere_function(x):
    return sum(xi**2 for xi in x)


# Hill Climbing
def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    current_point = tuple([random.uniform(b[0], b[1]) for b in bounds])
    current_value = func(current_point)
    step_size = 0.1

    for iteration in range(iterations):
        x, y = current_point
        neighbors = [
            (x + step_size, y),
            (x - step_size, y),
            (x, y + step_size),
            (x, y - step_size),
        ]

        # Пошук найкращого сусіда
        next_point = None
        next_value = np.inf

        for neighbor in neighbors:
            # Якщо сусід поза межами bounds - пропустити
            if (
                neighbor[0] < bounds[0][0]
                or neighbor[0] > bounds[0][1]
                or neighbor[1] < bounds[1][0]
                or neighbor[1] > bounds[1][1]
            ):
                continue
            value = func(neighbor)
            if value < next_value:
                next_point = neighbor
                next_value = value

        # Якщо не вдається знайти кращого сусіда
        # або різниця значень функції у поточній точці і сусідній менше epsilon — зупиняємось
        if next_value >= current_value or abs(current_value - next_value) < epsilon:
            break

        # Переходимо до кращого сусіда
        current_point, current_value = next_point, next_value

    return current_point, current_value


# Random Local Search
def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    current_point = tuple([random.uniform(b[0], b[1]) for b in bounds])
    current_value = func(current_point)
    step_size = 1

    for iteration in range(iterations):
        x, y = current_point

        # Якщо сусід не виходить за межі bounds
        if (
            x - step_size >= bounds[0][0]
            or x + step_size <= bounds[0][1]
            or y - step_size >= bounds[1][0]
            or y + step_size <= bounds[1][1]
        ):
            new_point = (
                x + random.uniform(-step_size, step_size),
                y + random.uniform(-step_size, step_size),
            )
        else:
            break

        new_value = func(new_point)

        # Якщо не вдається знайти кращого сусіда
        # або різниця значень функції у поточній точці і сусідній менше epsilon — зупиняємось
        if new_value >= current_value or abs(current_value - new_value) < epsilon:
            break

        # Переходимо до кращого сусіда
        current_point, current_value = new_point, new_value

    return current_point, current_value


# Simulated Annealing
def simulated_annealing(
    func, bounds, iterations=1000, temp=1000, cooling_rate=0.95, epsilon=1e-6
):
    current_solution = tuple([random.uniform(b[0], b[1]) for b in bounds])
    current_energy = func(current_solution)
    i = 1

    while temp > epsilon and i <= iterations:
        x, y = current_solution
        step_size = 1
        new_x = x + random.uniform(-step_size, step_size)
        new_y = y + random.uniform(-step_size, step_size)

        # Якщо нове рішення не виходить за межі bounds
        if (
            new_x - step_size >= bounds[0][0]
            or new_x + step_size <= bounds[0][1]
            or new_y - step_size >= bounds[1][0]
            or new_y + step_size <= bounds[1][1]
        ):
            new_solution = (new_x, new_y)
        else:
            break
        new_energy = func(new_solution)
        delta_energy = new_energy - current_energy

        if delta_energy < epsilon or random.random() < math.exp(-delta_energy / temp):
            current_solution = new_solution
            current_energy = new_energy

        temp *= cooling_rate
        i += 1

    return current_solution, current_energy


if __name__ == "__main__":
    # Межі для функції
    bounds = [(-5, 5), (-5, 5)]

    # Виконання алгоритмів
    print("Hill Climbing:")
    hc_solution, hc_value = hill_climbing(sphere_function, bounds)
    print("Розв'язок:", hc_solution, "Значення:", hc_value)

    print("\nRandom Local Search:")
    rls_solution, rls_value = random_local_search(sphere_function, bounds)
    print("Розв'язок:", rls_solution, "Значення:", rls_value)

    print("\nSimulated Annealing:")
    sa_solution, sa_value = simulated_annealing(sphere_function, bounds)
    print("Розв'язок:", sa_solution, "Значення:", sa_value)
