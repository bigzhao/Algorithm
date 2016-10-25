# coding=utf-8
from numpy import cos
# z = ((1*cos((1+1)*x+1))+(2*cos((2+1)*x+2))+(3*cos((3+1)*x+3)) +
#     (4*cos((4+1)*x+4))+(5*cos((5+1)*x+5)))*((1*cos((1+1)*y+1)) +
#     (2*cos((2+1)*y+2))+(3*cos((3+1)*y+3))+(4*cos((4+1)*y+4)) +
#     (5*cos((5+1)*y+5)))
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math

def draw_multimodal_function():
    with plt.style.context('dark_background'):
        fig = plt.figure()
        ax = Axes3D(fig)
        x = np.arange(-4, 4, 0.25)
        y = np.arange(-4, 4, 0.25)
        x, y = np.meshgrid(x, y)
        z = ((1*cos((1+1)*x+1))+(2*cos((2+1)*x+2))+(3*cos((3+1)*x+3)) +
            (4*cos((4+1)*x+4))+(5*cos((5+1)*x+5)))*((1*cos((1+1)*y+1)) +
            (2*cos((2+1)*y+2))+(3*cos((3+1)*y+3))+(4*cos((4+1)*y+4)) +
            (5*cos((5+1)*y+5)))
        # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
        ax.plot_surface(x, y, z, rstride=1, cstride=1,  cmap="rainbow", linewidth=0, antialiased=False)
        plt.show()
# ----------------------------------------------------------------------


def griewank_func(chromosome):
    """F6 Griewank's function
    multi-modal, symmetric, inseparable"""
    part1 = 0
    for i in range(len(chromosome)):
        part1 += chromosome[i]**2
    part2 = 1
    for i in range(len(chromosome)):
        part2 *= math.cos(float(chromosome[i]) / math.sqrt(i+1))
    return 1 + (float(part1)/4000.0) - float(part2)


def draw_griewank_func():
    with plt.style.context('dark_background'):
        fig = plt.figure()
        ax = Axes3D(fig)
        x = np.arange(-8, 8, 0.25)
        y = np.arange(-8, 8, 0.25)
        x, y = np.meshgrid(x, y)
        [r,c] = np.shape(x)
        z = [[] for _ in range(r)]
        for i in range(r):
            for j in range(c):
                z[i].append(griewank_func([x[i][j], y[i][j]]))
        ax.plot_surface(x, y, np.array(z), rstride=1, cstride=1,  cmap="rainbow", linewidth=0, antialiased=False)
        plt.show()
# ----------------------------------------------------------------------


def schaffer_function(x, y):
    return 0.5 + (math.pow(np.sin(math.sqrt(x*x + y*y)), 2) - 0.5)/math.pow(1 + 0.001*(x*x + y*y), 2)


def draw_schaffer_function():
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(-8, 8, 0.25)
    y = np.arange(-8, 8, 0.25)
    x, y = np.meshgrid(x, y)
    [r,c] = np.shape(x)
    z = [[] for _ in range(r)]
    for i in range(r):
        for j in range(c):
            z[i].append(schaffer_function(x[i][j], y[i][j]))
    ax.plot_surface(x, y, np.array(z), rstride=1, cstride=1,  cmap="coolwarm", linewidth=0, antialiased=False)
    plt.show()


if __name__ == "__main__":
    draw_schaffer_function()