# coding:utf-8

def extract_min(Q, key):
    '''
    去找最小的键值
    :param Q: 剩下的顶点集
    :param key: 对应的键值
    :return: 最小键值的index
    '''
    min_index = 0
    for i in range(1, len(Q)):
        # print i, min_index, key[Q[5]]
        if key[Q[i]] < key[Q[min_index]]:
            min_index = i
    return Q[min_index]


def update(A, v, key, pi, Q):
    '''
    更新函数
    :param A: 邻接表
    :param v: 选中的index
    :param key: 键值集
    :param pi: 跟踪集 记录起点
    :param Q: 剩下的index集
    :return: 木有
    '''
    for u in A[v]:
        if u[0] in Q and u[1] < key[u[0]]:
            key[u[0]] = u[1]
            pi[u[0]] = v


def prim(A, num):
    '''
    Prim主算法
    :param A: 邻接表，列表表示，每个列表里面是元祖，元祖第一个元素是index，第二个是相应的权值
    :param num: 顶点的数量
    :return:
    '''
    # 初始化
    key = [float("inf") for _ in xrange(num)]
    pi = [0 for _ in xrange(num)]
    key[0] = 0
    Q = range(8)
    while Q:
        v = extract_min(Q, key)
        update(A, v, key, pi, Q)
        Q.remove(v)
        print v, key, Q
    print pi, sum(key)


if __name__ == "__main__":
    A = [[(1, 6), (2, 12)], [(0, 6), (2, 5), (3, 14), (5, 8)],
         [(0, 12), (1, 5), (6, 9), (4, 7)], [(1, 14), (5, 3)],
         [(2, 7), (5, 10), (7, 15)], [(1, 8), (3, 3), (4, 10)],
         [(2, 9)], [(4,15), ]]
    num = 8
    prim(A, num)
    # inf = float('inf')
    # print extract_min([3, 4, 5, 6, 7], [0, 6, 5, 14, 7, 8, 9, inf])