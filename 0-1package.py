# coding=utf-8

import sys
import random
import numpy as np
import matplotlib.pyplot  as plt

reload(sys)
sys.setdefaultencoding('utf-8')

WEIGHT = [220,208,198,192,180,180,165,162,160,158,
        155,130,125,122,120,118,115,110,105,101,
        100,100,98,96,95,90,88,82,80,77,
        75,73,70,69,66,65,63,60,58,56,
        50,30,20,15,10,8,5,3,1,1]
VALUE = [80,82,85,70,72,70,66,50,55,25,
        50,55,40,48,50,32,22,60,30,32,
        40,38,35,32,25,28,30,22,50,30,
        45,30,60,50,20,65,20,25,30,10,
        20,25,15,10,10,10,4,4,2,1]
MAX_WEIGHT = 1000


def object_function(x):
    # print x
    global MAX_WEIGHT, NUM, WEIGHT, VALUE
    values = 0
    weights = 0
    length = len(x)
    for i in range(length):
        if x[i]:
            weights += WEIGHT[i]
            values += VALUE[i]
    if MAX_WEIGHT < weights:
        return 0.1
    return values


def find_the_best(pop):
    best_individual = pop.individuals[0]
    best = best_individual.fitness
    for i in range(1, len(pop.individuals)):
        if pop.individuals[i].fitness > best:
            best_individual = pop.individuals[i]
            best = pop.individuals[i].fitness
    return [best_individual.x, best]


class Population(object):
    """种群"""
    def __init__(self, num, gene_sizes):
        self.individuals = [Individual(gene_sizes) for _ in range(num)]


class Individual(object):
    """个体"""
    def __init__(self, num):
        self.x = [random.randint(0, 1) for _ in range(num)]
        self.fitness = object_function(self.x)

    def cal_fitness(self):
        self.fitness = object_function(self.x)


def mutation(pop, pm):
    '''
    突变函数，随机概率,若小于pm则利用random产生一个int，在特定的位置上将0->1 or 1->0
    :param pop: 种群
    :param pm: 突变的概率
    :return:
    '''
    length = len(pop.individuals)
    gene_len = len(pop.individuals[0].x)
    for i in range(length):
        if random.random() < pm:
            # print "before:{}".format(pop.individuals[i].x)
            index = random.randint(0, gene_len-1)
            # print "index:{}".format(index)
            pop.individuals[i].x[index] = int(not pop.individuals[i].x[index])
            # print "after:{}".format(pop.individuals[i].x)
            pop.individuals[i].cal_fitness()


def crossover(pop, pc):
    '''
    交叉繁殖函数
    :param pop: 种群
    :param pc: 交叉繁殖的概率
    :return: 传引用，直接改变pop
    '''
    length = len(pop.individuals)
    gene_len = len(pop.individuals[0].x)
    # print gene_len
    for i in range(length-1):
        if random.random() < pc:
            cpoint = random.randint(0, gene_len)
            # print "cp{}".format(cpoint)
            temp1 = pop.individuals[i].x[:cpoint] + pop.individuals[i+1].x[cpoint:]
            temp2 = pop.individuals[i+1].x[:cpoint] + pop.individuals[i].x[cpoint:]
            pop.individuals[i].x = temp1[:]
            pop.individuals[i+1].x = temp2[:]
        pop.individuals[i].cal_fitness()
        pop.individuals[i+1].cal_fitness()


def selection(pop):
    '''
    物竞天择，采用转轮盘选择法
    :param pop: 种群
    :param fitvalues: 相对应的适合度
    :return: 传引用，直接改变pop
    '''
    new_pop = []
    length = len(pop.individuals)
    total_value = 0.0

    for i in pop.individuals:
        total_value += i.fitness
    cal_fitvalues = [pop.individuals[0].fitness/total_value,]
    for i in range(1, length):
        cal_fitvalues.append((pop.individuals[i].fitness/total_value+cal_fitvalues[i-1]))
    # print cal_fitvalues
    random_rate = [random.random() for _ in range(length)]
    random_rate.sort()
    # print random_rate

    j, k = 0, 0
    while k < length:
        if random_rate[k] <= cal_fitvalues[j]:
            new_pop.append(pop.individuals[j])
            k += 1
        else:
            j += 1
    pop = new_pop[:]
    # print pop


def ga(gene_sizes):
    '''
    主算法,流程：
    ①初始化种群
    ②选择
    ③交配繁衍
    ④突变
    ⑤循环
    :return:
    '''
    iteration_num = 3000
    individual_num = 100
    pm = 0.05
    pc = 0.8
    record = []
    pop = Population(individual_num, gene_sizes)
    best = find_the_best(pop)
    record.append(best)

    for i in pop.individuals:
        print i.x, object_function(i.x)
    print best[1]
    for _ in range(iteration_num):
        selection(pop)
        crossover(pop, pc)
        mutation(pop, pm)
        best = find_the_best(pop)
        if best[1] > record[-1][1]:
            record.append(best)

    best = record[-1]
    print "GA done"
    print "------------------------------------------------"
    print "the best individual is {}, it's fitness is {}".format(best[0], best[1])
    print "record {}".format(record)
    return record


def my_input():
    # global MAX_WEIGHT, WEIGHT, VALUE
    # MAX_WEIGHT = float(raw_input(u"请输入背包最大重量"))
    # num = int(raw_input(u"请输入物品个数"))
    # for i in range(num):
    #     WEIGHT.append(float(raw_input(u"请输入第{}件物品的重量".format(i+1))))
    #     VALUE.append(float(raw_input(u"请输入第{}件物品的价值".format(i+1))))

    # print object_function([random.randint(0,1) for _ in range(num)])
    ga(50)
    # print object_function([1, 1, 1, 0, 0, 1, 1, 1, 0, 0])
if __name__ == "__main__":
    my_input()