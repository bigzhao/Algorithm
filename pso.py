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


def objective_function(x):
    part_1, part_2 = 0, 1
    length = len(x)
    for i in range(length):
        part_1 += x[i] * x[i]
        part_2 *= np.cos(x[i]/np.sqrt(i+1))
    part_1 /= 1/4000.0
    return 1 + part_1 - part_2


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

def pso(w, c1, c2):
    numberParticles = 500
    numberIterations = 1000
    iteration = 0
    Dim = 50
    minX = -100.0
    maxX = 100.0
    bestGlobalPosition = []
    bestGlobalFitness = float("inf")
    minV = -1.0 * maxX
    maxV = maxX
    swarm = []
    # 实例化 Particle
    for i in range(numberParticles):
        randomPosition = []
        for j in range(Dim):
            lo = minX
            hi = maxX
            randomPosition.append((hi - lo) * random.random() + lo)
            # print randomPosition

        fitness = objective_function(randomPosition)
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
        # print swarm[i].fitness, bestGlobalFitness
    # w = 0.729
    # c1 = 1.49445
    # c2 = 1.49445

    record = [[], [], []]
    best_posx = []
    best_posy = []
    best_fitness = []
    iterator = []
    bestFitness = []
    #迭代 套公式求最优
    nums = 0
    for k in range(numberIterations):
    # while 1:
    #     sign = 0  # 标志 用来标记是否最优解发生了
        for i in range(numberParticles):
            record[0].append(swarm[i].position[0])
            record[1].append(swarm[i].position[1])
            record[2].append(swarm[i].fitness)
            currP = swarm[i]
            newVelocity = []
            for j in range(Dim):
                r1 = random.random()
                r2 = random.random()
                # print currP.velocity[j], currP.position[j], currP.bestPosition[j], bestGlobalPosition[j]
                # print r1, r2
                newVelocity.append((w * currP.velocity[j]) + (c1 * r1 * (currP.bestPosition[j] - currP.position[j])) + \
                                 (c2 * r2 * (bestGlobalPosition[j] - currP.position[j])))
                # print newVelocity
                if newVelocity[j] < minV:
                    newVelocity[j] = minV
                elif newVelocity[j] > maxV:
                    newVelocity[j] = maxV
            # print newVelocity
            currP.velocity = newVelocity[:]

            newPosition = []
            for j in range(Dim):
                # print 'currP.position[j] + newVelocity[j] = {}'.format(currP.position[j] + newVelocity[j])
                newPosition.append(currP.position[j] + newVelocity[j])
                if newPosition[j] < minX:
                    newPosition[j] = minX
                if newPosition[j] > maxX:
                    newPosition[j] = maxX
            # print newPosition
            currP.position = newPosition[:]
            # print newPosition
            newFitness = objective_function(newPosition)
            # print newPosition, newFitness
            if newFitness < currP.bestFitness:
                currP.bestFitness = newFitness
                currP.bestPosition = newPosition[:]
            if newFitness < bestGlobalFitness:
                sign = 1
                bestGlobalFitness = newFitness
                bestGlobalPosition = newPosition[:]
            best_posx.append(currP.bestPosition[0])
            best_posy.append(currP.bestPosition[1])
            best_fitness.append(currP.bestFitness)
        # iterator.append(k)
        bestFitness.append(bestGlobalFitness)
        # if sign:
        #     sign = 0
        # else:
        #     nums += 1
        # if nums > 100:
        #     break



    print "\nProcessing complete"
    print "Final best fitness = ", str(bestGlobalFitness)
    print "Best position/solution:"
    for i in range(Dim):
        print "x" + str(i) + " = ", str(bestGlobalPosition[i])+" "
    print "\nEnd PSO demonstration\n"
    # print iterator
    print bestFitness
    # plt.plot(iterator, bestFitness, 'b')
    # plt.show()
    return iterator, bestFitness, record, best_posx, best_posy, best_fitness

if __name__ == "__main__":
    # mpl.rcParams['font.sans-serif'] = ['SimHei']
    # matplotlib.rcParams['axes.unicode_minus'] = False
    # plt.style.use("ggplot")
    import time
    t = time.time()
    iterator1, bestFitness1, record, best_posx,best_posy,best_fitness = pso(0.729, 1.49445, 1.49445)
    print time.time()-t
    # iterator2, bestFitness2 = pso(-2, 1.49445, 1.49445)
    # iterator3, bestFitness3 = pso(-1.5, 1.49445, 1.49445)
    # iterator4, bestFitness4 = pso(-1, 1.49445, 1.49445)
    # iterator5, bestFitness5 = pso(-0.5, 1.49445, 1.49445)
    # plt.figure(1)  # 创建图表1
    # plt.plot(iterator1, bestFitness1, 'b')
    # plt.title('c=0.729')
    # fig = plt.figure(1, figsize=(14, 6))  # 创建图表2
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # # x = record[0]
    # # y = record[1]
    # # z = record[2]
    # # ax.scatter(x, y, z, c='b')
    # # plt.title('所有粒子走向图')
    #
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.scatter(best_posx, best_posy, best_fitness, c='r')
    # # plt.plot(iterator2, bestFitness2, 'b')
    # plt.title('局部最优解走向图')
    #
    #
    # plt.figure(3)  # 创建图表1
    # plt.plot(iterator3, bestFitness3, 'b')
    # plt.title('c=1.5')
    #
    #
    # plt.figure(4)  # 创建图表2
    # plt.plot(iterator4, bestFitness4, 'b')
    # plt.title('c=1')
    #
    #
    # plt.figure(5)  # 创建图表1
    # plt.plot(iterator5, bestFitness5, 'b')
    # plt.title('c=0.5')
    # fig2 = plt.figure(2)
    # ax = Axes3D(fig2)
    # plt.ion()
    # ax.set_xlim(-100, 100)
    # ax.set_ylim(-100, 100)
    # ax.set_zlim(0, 1)
    # for i in range(len(best_posx)):
    #     # y = np.random.random()
    #     ax.scatter(best_posx[i], best_posy[i], best_fitness[i], c='r')
    #     plt.pause(0.01)
    # plt.title("动态局部解走向图")
    # plt.show()



