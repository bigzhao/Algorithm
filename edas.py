# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
Dim = 100                                                # 问题维度
minX = -100.0
maxX = 100.0
shift_vector = [(maxX - minX)/2 * np.random.random() + minX/2 for _ in xrange(Dim)]                 # 位移向量
NP = 500
K = 250


def objective_function(x):
    part_1, part_2 = 0, 1
    after_shift_x = np.array(x) - np.array(shift_vector)
    # print after_shift_x
    for i in range(Dim):
        part_1 += after_shift_x[i] * after_shift_x[i]
        part_2 *= np.cos(after_shift_x[i]/np.sqrt(i+1))
    part_1 /= 1/4000.0
    return 1 + part_1 - part_2


class Individual:
    def __init__(self, position):
        self.position = position[:]
        self.fitness = objective_function(position)


def initialize_population(num):
    pop = []
    for i in range(num):
        position = [(maxX - minX) * np.random.random() + minX for _ in range(Dim)]
        pop.append(Individual(position))
    return pop


def get_k_best(pop, K):
    pop_bak = pop
    sorted(pop_bak, key = lambda x: x.fitness)
    return pop_bak[: K]


def generate_new_population(k_pop):
    new_pop = []
    temp = []
    for i in range(Dim):
        val = []
        for j in range(K):
            val.append(k_pop[j].position[i])
        temp.append((np.mean(val), np.std(val)))
    for i in range(NP):
        position = []
        for j in range(Dim):
            position.append(np.random.normal(temp[j][0], temp[j][1]))
        new_pop.append(Individual(position))
    return new_pop


def select(new, now):
    # combine = new + now
    for i in range(NP):
        if new[i].fitness < now[i].fitness:
            now[i] = new[i]


def find_the_best(pop):
    best = pop[0]
    for i in pop[1:]:
        if i.fitness < best.fitness:
            best = i
    return best


def EDAs():
    pop = initialize_population(NP)
    best_array = []
    for _ in range(200):
        k_pop = get_k_best(pop, K)
        # for i in range(Dim):
        new_pop = generate_new_population(k_pop)
        select(new_pop, pop)
        # print "best fitness = {:E}".format(find_the_best(pop).fitness)
        best_array.append(find_the_best(pop).fitness)

    print "\nProcessing complete"
    print "Final best fitness = {:E}".format(find_the_best(pop).fitness)
    print "Best position/solution:"
    for i in range(Dim):
        print "x" + str(i) + " = ", str(find_the_best(pop).position[i])+" "
    print "\nEnd PSO demonstration\n"
    return best_array

if __name__ == "__main__":
    best_array = EDAs()
    plt.plot(best_array)
    print shift_vector
    plt.show()