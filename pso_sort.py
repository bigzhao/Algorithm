# coding=utf-8
from __future__ import unicode_literals
import random
import matplotlib.pyplot  as plt

def objective_function(x):
    xx = []
    fitness = 0.0
    for i in range(1, 21):
        times = 0
        xx.append(x[20*(i-1):(i)*20])
        # print xx
        for j in x[20*(i-1):(i)*20]:
            # print j
            if (j < 1/4.0 and j >= 0) or (j<((i+1)/4.0) and j>=(i/4.0)):
                # print 'in'
                if (j<((i+1)/4.0) and j>=(i/4.0)):
                    times += 1
            else:
                fitness -= float("inf")
        if times>2:
            fitness -= float("inf")
            # print "times", times
        elif times == 2:
            fitness += 2
        elif times == 1:
            fitness += 1
        else:
            pass
    # print xx
    for i in range(3):
        times = 0
        for j in range(3):
            if xx[j][i] >= 1:
                times += 1
        if times > 2:
            fitness += 3
        elif times > 1:
            fitness += 2
        elif times > 0:
            fitness += 1
        else:
            pass
    return fitness




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
    numberParticles = 30
    numberIterations = 500
    iteration = 0
    Dim = 60
    minX = 0.0
    maxX = 1.0
    bestGlobalPosition = []
    bestGlobalFitness = float("-inf")
    minV = -1.0 * maxX
    maxV = maxX
    swarm = []
    # 实例化 Particle
    for i in range(numberParticles):
        randomPosition = []
        for j in range(Dim):
            lo = minX
            hi = maxX
            randomPosition.append((1/21.0) * random.random())
            # randomPosition.append(0.1)

            # print randomPosition

        fitness = objective_function(randomPosition)
        randomVelocity = []
        for j in range(Dim):
            lo = -1.0 * abs(maxX - minX)
            hi = abs(maxX - minX)
            randomVelocity.append((hi - lo) * random.random() + lo)
        swarm.append(Particle(position=randomPosition, fitness=fitness, velocity=randomVelocity,
                              bestPosition=randomPosition, bestFitness=fitness))
        if swarm[i].fitness > bestGlobalFitness:
            bestGlobalFitness = swarm[i].fitness
            bestGlobalPosition = swarm[i].position
        # print bestGlobalFitness
        # print swarm[i].fitness, bestGlobalFitness
    # w = 0.729
    # c1 = 1.49445
    # c2 = 1.49445

    iterator = []
    bestFitness = []
    #迭代 套公式求最优
    for k in range(numberIterations):
        for i in range(numberParticles):
            currP = swarm[i]
            newVelocity = []
            for j in range(Dim):
                r1 = random.random()
                r2 = random.random()
                # print currP.velocity[j], currP.position[j], currP.bestPosition[j], bestGlobalPosition[j]
                # print r1, r2
                # print j,bestGlobalPosition
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
            if newFitness > currP.bestFitness:
                currP.bestFitness = newFitness
                currP.bestPosition = newPosition[:]
            if newFitness > bestGlobalFitness:
                bestGlobalFitness = newFitness
                bestGlobalPosition = newPosition[:]

        iterator.append(k)
        bestFitness.append(bestGlobalFitness)


    print "\nProcessing complete"
    print "Final best fitness = ", str(bestGlobalFitness)
    print "Best position/solution:"
    for i in range(Dim):
        print "x" + str(i) + " = ", str(bestGlobalPosition[i])+" "
    print "\nEnd PSO demonstration\n"
    print iterator
    print bestFitness
    # plt.plot(iterator, bestFitness, 'b')
    # plt.show()
    # return iterator, bestFitness

if __name__ == "__main__":
    pso( w=0.729, c1=1.49445, c2=1.49445)
    # print objective_function([0 for _ in range(100)])


