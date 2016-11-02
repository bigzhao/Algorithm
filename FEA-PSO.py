# coding=utf-8
from __future__ import unicode_literals
import random
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
from pylab import mpl
from mpl_toolkits.mplot3d import Axes3D
from numpy import cos

# 问题的维度
NUM = 500
GLOBAL_SOLUTION = [400 * random.random() - 200 for _ in xrange(NUM)]


def objective_function(x):
    part_1, part_2 = 0, 1
    length = len(x)
    for i in range(length):
        part_1 += x[i] * x[i]
        part_2 *= np.cos(x[i]/np.sqrt(i+1))
    part_1 /= 1/4000.0
    return 1 + part_1 - part_2


def rebuild(sub_index, sub_solution, global_solution):
    '''
    与全局解重组得到可以计算适应度的solution
    :param sub_index: 亚种群占有的维度
    :param sub_solution: 部分解
    :param global_solution: 全局解
    :return:
    '''
    index = 0
    rebuild_solution = []
    for i in xrange(NUM):
        if i in sub_index:
            rebuild_solution.append(sub_solution[index])
            index += 1
        else:
            rebuild_solution.append(global_solution[i])
    return rebuild_solution


def initialize_subpops(l):
    '''
    初始化并划分结构这里使用cs-l结构
    :param l: cs-l的l
    :return: 亚种群列表
    '''
    # 这里使用CS-l结构
    sub_index = [[j + i for j in range(l + 1)] for i in range(NUM - l)]
    sub_populations = []

    # 随机初始化全局解
    number_particles = 10
    numberIterations = 20

    # 实例化 Particle
    for s in sub_index:
        #初始条件
        Dim = len(s)
        minX = -100.0
        maxX = 100.0
        lo = -1.0 * abs(maxX - minX)
        hi = abs(maxX - minX)
        bestGlobalPosition = []
        bestGlobalFitness = float("inf")

        minV = -1.0 * maxX
        maxV = maxX
        swarm = []

        for i in range(number_particles):
            randomPosition = []
            for j in range(Dim):
                lo = minX
                hi = maxX
                randomPosition.append((hi - lo) * random.random() + lo)
                # print randomPosition

            fitness = objective_function(rebuild(s, randomPosition, GLOBAL_SOLUTION))
            randomVelocity = []
            for j in range(Dim):
                lo = -1.0 * abs(maxX - minX)
                hi = abs(maxX - minX)
                randomVelocity.append((hi - lo) * random.random() + lo)
            swarm.append(Particle(position=randomPosition, fitness=fitness, velocity=randomVelocity,
                                  bestPosition=randomPosition, bestFitness=fitness))
            if swarm[i].fitness < bestGlobalFitness:
                bestGlobalFitness = swarm[i].fitness
                bestGlobalPosition = swarm[i].position

        sub_populations.append((swarm[:], bestGlobalPosition[:], bestGlobalFitness))
        # print (swarm[:], bestGlobalPosition[:], bestGlobalFitness)
    return sub_index, sub_populations


class Particle(object):
    def __init__(self, position, fitness, velocity, bestPosition, bestFitness):
        self.position = position[:]
        self.fitness = fitness
        self.velocity = velocity[:]
        self.bestPosition = bestPosition[:]
        self.bestFitness = bestFitness

    def to_string(self):
        s = ""
        s += "==========================\n"
        s += "Position: "
        for i in range(len(self.position)):
            s += str(self.position[i]) + " "
        s += "\n"
        s += "Fitness = " + str(self.fitness) + "\n"
        s += "Velocity: "
        for i in range(len(self.velocity)):
            s += str(self.velocity[i]) + " "
        s += "\n"
        s += "Best Position: "
        for i in range(len(self.bestPosition)):
            s += str(self.bestPosition[i]) + " "
        s += "\n"
        s += "Best Fitness = " + str(self.bestFitness) + "\n"
        s += "==========================\n"
        return s


def pso(index, sub_population):
    '''
    亚种群的pso优化
    :param index: sub_index的元素 标记亚种群的维度
    :param sub_population: 亚种群，0：粒子群，1：亚种群全局最优值，2：亚种群最优FItness
    :return: 元组 0：粒子群，1：亚种群全局最优值，2：亚种群最优FItness
    '''
    swarm = sub_population[0]
    best_global_position = sub_population[1]
    best_global_fitness = sub_population[2]
    number_particles = 10
    number_iterations = 20
    minX = -100.0
    maxX = 100.0
    minV = -1.0 * maxX
    maxV = maxX
    w = 0.729
    c1 = 1.49445
    c2 = 1.49445
    Dim = len(index)
    for k in range(number_iterations):
        for i in range(number_particles):
            currP = swarm[i]
            new_velocity = []
            for j in range(Dim):
                r1 = random.random()
                r2 = random.random()
                new_velocity.append((w * currP.velocity[j]) + (c1 * r1 * (currP.bestPosition[j] - currP.position[j])) + \
                                 (c2 * r2 * (best_global_position[j] - currP.position[j])))

                if new_velocity[j] < minV:
                    new_velocity[j] = minV
                elif new_velocity[j] > maxV:
                    new_velocity[j] = maxV
            currP.velocity = new_velocity[:]

            new_position = []
            for j in range(Dim):
                new_position.append(currP.position[j] + new_velocity[j])
                if new_position[j] < minX:
                    new_position[j] = minX
                if new_position[j] > maxX:
                    new_position[j] = maxX
            currP.position = new_position[:]
            new_fitness = objective_function(rebuild(index, new_position, GLOBAL_SOLUTION))
            if new_fitness < currP.bestFitness:
                currP.bestFitness = new_fitness
                currP.bestPosition = new_position[:]
            if new_fitness < best_global_fitness:
                best_global_fitness = new_fitness
                best_global_position = new_position[:]
    return swarm[:], best_global_position[:], best_global_fitness


def competition(subpopulations, overlap_dict):
    global GLOBAL_SOLUTION
    order_seq = range(NUM)
    random.shuffle(order_seq)
    best_fit = objective_function(GLOBAL_SOLUTION)
    for i in order_seq:
        overlap_index = overlap_dict[i]
        random.shuffle(overlap_dict[i])
        for j in overlap_index:
            s = subpopulations[j[0]]
            temp = GLOBAL_SOLUTION[:]
            temp[i] = s[1][i - j[0]]    # s[1]是当前的bestposition，i-j[0]是相对的位置
            fitness = objective_function(temp)
            if fitness < best_fit:
                GLOBAL_SOLUTION[i] = s[1][i - j[0]]


def find_the_most_partical(swarms):
    # print swarms
    worst = swarms[0]
    for s in swarms[1:]:
        if worst.fitness < s.fitness:
            worst = s
    return worst


def sharing(sub_index, subpopulations):
    order_seq = range(len(sub_index))
    random.shuffle(order_seq)
    for i in order_seq:
        index = sub_index[i]
        worst_partical = find_the_most_partical(subpopulations[index[0]][0])
        worst_partical.position = [GLOBAL_SOLUTION[j] for j in index]
        worst_partical.fitness = objective_function(GLOBAL_SOLUTION)




def fea_pso():
    """FEA版本的PSO"""
    sub_index, subpopulations = initialize_subpops(10)
    overlap_dict = {}
    for i in range(NUM):
        overlap_dict[i] = []
        for s in sub_index:
            if i in s:
                overlap_dict[i].append(s)
    # print subpopulations[0]
    best_fitness = [objective_function(GLOBAL_SOLUTION)]
    record = 0
    while 1:
        # 亚种群优化
        for i in xrange(len(sub_index)):
            subpopulations[i] = pso(sub_index[i], subpopulations[i])
        # 竞争
        competition(subpopulations, overlap_dict)
        #分享
        sharing(sub_index, subpopulations)
        now_fitness = objective_function(GLOBAL_SOLUTION)
        if now_fitness == best_fitness[-1]:
            record += 1
        else:
            record = 0
        best_fitness.append(now_fitness)
        print now_fitness
        if 10 < record:
            break

    print GLOBAL_SOLUTION, objective_function(GLOBAL_SOLUTION)
    return best_fitness
    # return subpopulations

if __name__ == "__main__":
    # print objective_function((0,0,0))
    # subpopulations = fea_pso()
    # print subpopulations[0]
    import time
    t = time.time()
    best_fitness = fea_pso()
    print time.time() - t
    plt.plot(best_fitness, 'r')
    plt.show()

