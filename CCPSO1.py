# coding=utf-8
import numpy as np

Dim = 20                                                # 问题维度
S = 2                                                   # 分组的维度长度
minX = -100.0
maxX = 100.0
minV = -1.0 * maxX
maxV = maxX
W = 0.729                                               # PSO 的W
c1 = 1.49445                                            # PSO 的 C1
c2 = 1.49445
numberParticles = 10                                    # 粒子的数量
numberIterations = 20                                   # 迭代的次数
BEST_GLOBAL_POSITION = []                               # 全局最优
BEST_GLOBAL_FITNESS = float("inf")                      # 最好适应度
shift_vector = [(maxX - minX)/2 * np.random.random() + minX/2 for _ in xrange(Dim)]                 # 位移向量

def objective_function(x):
    part_1, part_2 = 0, 1
    after_shift_x = np.array(x) - np.array(shift_vector)
    # print after_shift_x
    for i in range(Dim):
        part_1 += after_shift_x[i] * after_shift_x[i]
        part_2 *= np.cos(after_shift_x[i]/np.sqrt(i+1))
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


class Swarm(object):
    def __init__(self, best_global_position, best_global_fitness):
        self.best_global_position = best_global_position[:]
        self.best_global_fitness = best_global_fitness


def build_solution(index, indices, position):
    '''
    将分组里面的其余位置用全局最优去替换
    :param index: 第几组
    :param indices: 随机产生的下标
    :param position: 要替换的值
    :return:
    '''
    solution = BEST_GLOBAL_POSITION[:]
    for i in indices[index * S: (index + 1) * S]:
        solution[i] = position[i]
    return solution


def construct_sub_swarms():
    """创建分组"""
    swarms = []
    for i in xrange(Dim / S):
        swarms.append(Swarm(best_global_position=BEST_GLOBAL_POSITION,
                            best_global_fitness=objective_function(BEST_GLOBAL_POSITION)))
    return swarms


def initialize_swarm():
    """对PSO的大种群进行初始化"""
    global BEST_GLOBAL_FITNESS, BEST_GLOBAL_POSITION
    swarm = []
    # 实例化 Particle
    for i in range(numberParticles):
        random_position = []
        for j in range(Dim):
            lo = minX
            hi = maxX
            random_position.append((hi - lo) * np.random.random() + lo)

        fitness = objective_function(random_position)
        random_velocity = []
        for j in range(Dim):
            lo = -1.0 * abs(maxX - minX)
            hi = abs(maxX - minX)
            random_velocity.append((hi - lo) * np.random.random() + lo)
        swarm.append(Particle(position=random_position, fitness=fitness, velocity=random_velocity,
                              bestPosition=random_position, bestFitness=fitness))
        if swarm[i].fitness < BEST_GLOBAL_FITNESS:
            BEST_GLOBAL_FITNESS = swarm[i].fitness
            BEST_GLOBAL_POSITION = swarm[i].position
    return swarm


def construct_with_weighting(w, indices):
    '''
    将全局最优跟产生的权重W相×，注意这里W的长度是组的个数
    :param w: 权重向量
    :param indices: 随机打乱的下标
    :return:
    '''
    global BEST_GLOBAL_POSITION
    new_solution = BEST_GLOBAL_POSITION[:]
    j = 0
    count = 0
    for i in indices:
        new_solution[i] *= w[j]
        count += 1
        if Dim / S <= count:
            j += 1
            count = 0
    return new_solution


def adaptive_weighting(indices):
    '''
    利用PSO去优化权重，细调最优解
    :param indices: 随机的下标
    :return: 最优的权重向量
    '''
    global BEST_GLOBAL_POSITION
    best_global_position = []
    best_global_fitness = float("inf")
    swarm = []
    max_w = []
    for i in range(Dim / S):
        max_val = abs(BEST_GLOBAL_POSITION[indices[i * S]])
        for j in indices[i * S + 1: (i + 1) * S]:
            if max_val < abs(BEST_GLOBAL_POSITION[j]):
                max_val = abs(BEST_GLOBAL_POSITION[j])
        max_w.append(100.0 / max_val if max_val != 0 else 1)
    # print max_w
    # 实例化 Particle
    for i in range(numberParticles):
        random_position = []
        for j in range(Dim / S):
            lo = -1 * max_w[j]
            hi = max_w[j]
            random_position.append((hi - lo) * np.random.random() + lo)

        fitness = objective_function(construct_with_weighting(random_position, indices))
        random_velocity = []
        for j in range(Dim / S):
            lo = -1.0 * abs(maxX - minX)
            hi = abs(maxX - minX)
            random_velocity.append((hi - lo) * np.random.random() + lo)
        swarm.append(Particle(position=random_position, fitness=fitness, velocity=random_velocity,
                              bestPosition=random_position, bestFitness=fitness))
        if swarm[i].fitness < best_global_fitness:
            best_global_fitness = swarm[i].fitness
            best_global_position = swarm[i].position

    for i in range(numberIterations):
        for j in range(numberParticles):
            curr_p = swarm[j]
            for k in range(Dim / S):
                r1 = np.random.random()
                r2 = np.random.random()
                new_velocity = (W * curr_p.velocity[k]) + (c1 * r1 * (curr_p.bestPosition[k] - curr_p.position[k])) + \
                               (c2 * r2 * (best_global_position[k] - curr_p.position[k]))

                if new_velocity < minV:
                    new_velocity = minV
                elif new_velocity > maxV:
                    new_velocity = maxV
                curr_p.velocity[k] = new_velocity
                new_position = curr_p.position[k] + new_velocity
                if new_position < minX:
                    new_position = minX
                if new_position > maxX:
                    new_position = maxX
                curr_p.position[k] = new_position
            new_fitness = objective_function(construct_with_weighting(curr_p.position, indices))
            if new_fitness < curr_p.bestFitness:
                curr_p.bestFitness = new_fitness
                curr_p.bestPosition = curr_p.position[:]
            if new_fitness < best_global_fitness:
                best_global_fitness = new_fitness
                best_global_position = curr_p.position[:]
    return best_global_position


def ccpso():
    """CCPSO主算法，流程大概是初始化->循环->自适应权重->终止条件满足退出"""
    global BEST_GLOBAL_POSITION, BEST_GLOBAL_FITNESS
    swarm = initialize_swarm()
    indices = [i for i in range(Dim)]
    #迭代 套公式求最优
    for _ in range(10):
        np.random.shuffle(indices)
        sub_swarms = construct_sub_swarms()
        for i in range(Dim / S):
            for j in xrange(numberParticles):
                curr_p = swarm[j]
                if objective_function(build_solution(i, indices, curr_p.position)) < \
                        sub_swarms[i].best_global_fitness:
                    sub_swarms[i].best_global_position = curr_p.position[:]
                if objective_function(build_solution(i, indices, sub_swarms[i].best_global_position)) < \
                        BEST_GLOBAL_FITNESS:
                    BEST_GLOBAL_FITNESS = objective_function(
                        build_solution(i, indices, sub_swarms[i].best_global_position))
                    for index in indices[i * S: (i + 1) * S]:
                        BEST_GLOBAL_POSITION[index] = sub_swarms[i].best_global_position[index]

                for j in range(numberIterations):
                    for k in range(numberParticles):
                        curr_p = swarm[k]
                        for m in indices[i * S: (i + 1) * S]:
                            r1 = np.random.random()
                            r2 = np.random.random()
                            new_velocity = (W * curr_p.velocity[m]) + \
                                           c1 * r1 * (curr_p.bestPosition[m] - curr_p.position[m]) + \
                                           c2 * r2 * (sub_swarms[i].best_global_position[m] - curr_p.position[m])
                            if new_velocity < minV:
                                new_velocity = minV
                            elif new_velocity > maxV:
                                new_velocity = maxV
                            curr_p.velocity[m] = new_velocity
                            new_position = curr_p.position[m] + new_velocity
                            if new_position < minX:
                                new_position = minX
                            if new_position > maxX:
                                new_position = maxX
                            curr_p.position[m] = new_position
                        curr_p.fitness = objective_function(curr_p.position)
                        if curr_p.fitness < curr_p.bestFitness:
                            curr_p.bestFitness = curr_p.fitness
                            curr_p.bestPosition = curr_p.position[:]
                        if curr_p.fitness < sub_swarms[i].best_global_fitness:
                            sub_swarms[i].best_global_fitness = curr_p.fitness
                            sub_swarms[i].best_global_position = curr_p.position[:]
        w = adaptive_weighting(indices)
        new_solution = construct_with_weighting(w, indices)
        if objective_function(new_solution) < BEST_GLOBAL_FITNESS:
            BEST_GLOBAL_FITNESS = objective_function(new_solution)
            BEST_GLOBAL_POSITION = new_solution[:]
        print "{:.2E}".format(BEST_GLOBAL_FITNESS)



    print "\nProcessing complete"
    print "Final best fitness = ", str(BEST_GLOBAL_FITNESS)
    print "Best position/solution:"
    for i in range(Dim):
        print "x" + str(i) + " = ", str(BEST_GLOBAL_POSITION[i])+" "
    print "\nEnd PSO demonstration\n"

if __name__ == "__main__":
    ccpso()


