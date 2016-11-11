# coding=utf-8
"""
Author Name: bigzhao
Email      : tandazhao@email.szu.edu.cn
"""
from __future__ import unicode_literals
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl

Dim = 2
MAX = 100
MIN = -100
F = 0.5
CR = 0.3
NUM = 100


def objective_function(x):
    part_1, part_2 = 0, 1
    for i in range(Dim):
        part_1 += x[i] * x[i]
        part_2 *= np.cos(x[i]/np.sqrt(i+1))
    part_1 /= 1/4000.0
    return 1 + part_1 - part_2


class Individual(object):
    def __init__(self, x, fitness):
        self.x = x[:]
        self.fitness = fitness


def initiate_population(num):
    population = []
    for _ in xrange(num):
        x = ((MAX-MIN)*np.random.rand(2) + MIN)
        fitness = objective_function(x)
        population.append(Individual(x, fitness))
    return population


def mutation(population):
    h = []
    for i in xrange(NUM):
        array = range(NUM)
        np.random.shuffle(array)
        position = []
        for j in xrange(Dim):
            new_position = population[array[0]].x[j] + F * (population[array[1]].x[j] - population[array[2]].x[j])
            if new_position < MIN:
                new_position = MIN
            if new_position > MAX:
                new_position = MAX
            position.append(new_position)
        h.append(Individual(position, objective_function(position)))
    return h


def crossover(population, mutation_population):
    new_population = []
    for i in xrange(NUM):
        position = [mutation_population[i].x[j] if np.random.random() < CR else population[i].x[j] for j in xrange(Dim)]
        new_population.append(Individual(position, objective_function(position)))
    return new_population


def select(new_population, population):
    for i in xrange(NUM):
        if new_population[i].fitness < population[i].fitness:
            population[i] = new_population[i]


def find_the_best(population):
    """找到种群里适应度最好的个体"""
    best = population[0]
    for i in population[1:]:
        if i.fitness < best.fitness:
            best = i
    return best


def de():
    population = initiate_population(NUM)
    last_fitness = find_the_best(population)
    record = 0
    fitness_record = []
    while 1:
        mutution_population = mutation(population)
        new_population = crossover(population, mutution_population)
        select(new_population, population)
        new_fitness = find_the_best(population).fitness
        if new_fitness < last_fitness:
            record = 0
            last_fitness = new_fitness
        else:
            record += 1
        if 15 <= record:
            break
        fitness_record.append(last_fitness)
    return fitness_record


if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.style.use("ggplot")
    fitness_record = de()
    de = plt.semilogy(fitness_record, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend(de, ("DE",))
    plt.title("差分进化算法")
    plt.show()
