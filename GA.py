# coding=utf-8
import sys
import random
import matplotlib.pyplot  as plt

reload(sys)
sys.setdefaultencoding('utf-8')


def b2d(x):
    temp = 0
    for i in range((len(x))):
        temp += (2 ** i) * x[i]
    return temp


def d2b(x):
    temp = [int(i) for i in bin(x)[2:]]
    while(len(temp) < 10):
        temp.insert(0, 0)
    return temp


def object_function(x, y):
    x = b2d(x) * 50 / 1024.0                 #转换
    y = b2d(y) * 50 / 1024.0
    return x * x + 1 + y * y


def find_the_best(pop):
    best_individual = pop.individuals[0]
    best = best_individual.fitness
    for i in range(1, len(pop.individuals)):
        if pop.individuals[i].fitness < best:
            best_individual = pop.individuals[i]
            best = pop.individuals[i].fitness
    return [(b2d(best_individual.x[0]) *50/ 1024.0, b2d(best_individual.x[1]) * 50 / 1024.0), best]


class Population(object):
    """种群"""
    def __init__(self, num):
        self.pmax = 10
        self.pmin = 0
        self.individuals = [Individual(self.pmax, self.pmin) for _ in range(num)]


class Individual(object):
    """个体"""
    def __init__(self, vmax, vmin):
        self.x = [d2b(int(random.random() * 1023)), d2b(int(random.random() * 1023))]
        # print self.x
        # self.x = d2b(self.x)
        # self.y = d2b(self.y)
        # print self.x
        # self.y = random.random() * (vmax-vmin) + vmin
        self.fitness = object_function(self.x[0], self.x[1])

    def cal_fitness(self):
        self.fitness = object_function(self.x[0], self.x[1])


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
        for j in range(2):
            if random.random() < pm:
                # print "before:{}".format(pop.individuals[i].x)
                index = random.randint(0, gene_len-1)
                # print "index:{}".format(index)
                pop.individuals[i].x[j][index] = int(not pop.individuals[i].x[j][index])
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
    for i in range(length-1):
        for j in range(2):
            if random.random() < pc:
                cpoint = random.randint(0, gene_len)
                # print "cp{}".format(cpoint)
                temp1 = pop.individuals[i].x[j][:cpoint] + pop.individuals[i+1].x[j][cpoint:]
                temp2 = pop.individuals[i+1].x[j][:cpoint] + pop.individuals[i].x[j][cpoint:]
                pop.individuals[i].x[j] = temp1[:]
                pop.individuals[i + 1].x[j] = temp2[:]
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


def ga():
    '''
    主算法,流程：
    ①初始化种群
    ②选择
    ③交配繁衍
    ④突变
    ⑤循环
    :return:
    '''
    iteration_num = 1000
    individual_num = 100
    pm = 0.05
    pc = 0.8
    record = []
    times = 0
    sign = 0
    pop = Population(individual_num)
    # best = [b2d(pop.individuals[0].x[0]) * 10 /1024.0, b2d(pop.individuals[0].x[1]) * 10 /1024.0]
    nums = 0
    while 1:
        selection(pop)
        crossover(pop, pc)
        mutation(pop, pm)
        best = find_the_best(pop)
        # temp, b = find_the_best(pop)
        # if abs(temp[0] - best[0]) < 0.1 and abs(temp[0][1] - best[0][1]) < 0.1:
        #     if times >= 20:
        #         break
        #     elif times:
        #         times += 1
        #     else:
        #         times = 1
        # else:
        #     if times:
        #         times = 0
        if len(record) == 0 or best[1] < record[-1][1]:
            record.append(best)
            nums = 0
        else:
            nums += 1
        if nums > 100:
            break

    best = record[-1]
    print "GA done"
    print "------------------------------------------------"
    print "the best individual is {}, it's fitness is {}".format(best[0] , best[1])
    print "record {}".format(record)
    return record





if __name__ == "__main__":
    # seletion([1,2,3,4,5], [1 ,3, 0, 2, 4] )
    # pop = [[1, 0, 0, 0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0, 1, 1, 1, 1]]
    # print pop
    # crossover(pop, 0.95)
    # print pop
    # while True:
    #     mutation(pop, 0.005)
    res = ga()
    fitness = [res[i][1] for i in range(len(res))]
    iteration = [i for i in range(len(res))]
    # print d2b(500)
    plt.figure(1)  # 创建图表1
    plt.plot(iteration, fitness, 'b')
    plt.title('GA')
    plt.show()