# coding:utf-8
"""
bigzhao, 2016/10/27
"""

def dynamic_programming_solve_lcs(x, y):
    """use dynamic programming to solve the LCS problem time: O(mn), space: O(mn)"""
    c = [[0 for _ in xrange(len(x)+1)] for _ in xrange(len(y)+1)]
    b = [['' for _ in xrange(len(x)+1)] for _ in xrange(len(y)+1)]
    for i in xrange(1, len(y) + 1):
        for j in xrange(1, len(x) + 1):
            if y[i-1] == x[j-1]:
                c[i][j] = c[i-1][j-1] + 1
                b[i][j] = 'q'
            elif c[i-1][j] > c[i][j-1]:
                c[i][j] = c[i-1][j]
                b[i][j] = 'w'
            else:
                c[i][j] = c[i][j-1]
                b[i][j] = 'a'
    return c, b


def memoization_solve_lcs(x, y, j, i, c):
    """use memoization to solve the LCS problem time: O(mn), space: O(mn)"""
    if j < 0 or i < 0:
        return 0
    if c[i][j] == "*":
        if x[j] == y[i]:
            c[i][j] = memoization_solve_lcs(x, y, j-1, i-1, c) + 1
        else:
            c[i][j] = max(memoization_solve_lcs(x, y, j-1, i, c), memoization_solve_lcs(x, y, j, i-1, c))
    return c[i][j]

A, B = "ABCBDAB", "BDCABA"
c, b = dynamic_programming_solve_lcs("ABCBDAB", "BDCABA")
print "The length of the longest common substring is {}".format(c[-1][-1])
i, j = len(b[0])-1, len(b)-1
lcs = []
while b[j][i] != '':
    if b[j][i] == "q":
        lcs.append(A[i-1])
        i -= 1
        j -= 1
    elif b[j][i] == "w":
        j -= 1
    elif b[j][i] == "a":
        i -= 1
    else:
        pass
print "The longest common substring is \"", "".join(lcs[::-1]),"\""
# c = [["*" for _ in range(7)] for _ in range(6)]
# print c
# print memoization_solve_lcs("ABCBDAB", "BDCABA", 6, 5, c)