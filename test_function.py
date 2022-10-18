# ！usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2021/12/6 10:44
# @Author : LucXiong
# @Project : Model
# @File : test_func.py

"""
https://blog.csdn.net/miscclp/article/details/38102831
"""
import math
import random
from scipy.stats import norm
# Unimodal test functions
def fu1(x):
    # Sphere function
    # min is 0 at [0 for i in range(len(x)]
    s1 = 0
    for i in range(len(x)):
        s1 += (x[i] ** 2)
    return s1

def fu2(x):
    '''
    min is 0 at [0 for i in range(len(x)]
    :param x: xi ∈ [-10, 10]
    :return:
    '''

    s1 = 0
    s2 = 1
    for i in range(len(x)):
        s1 += abs(x[i])
        s2 *= abs(x[i])
    result = s1 + s2
    return result

def fu3(x):
    # min is 0 at [0 for i in range(len(x)]
    s1 = 0
    for i in range(len(x)):
        s2 = 0
        for j in range(i):
            s2 += abs(x[i])
        s1 += s2 ** 2
    result = s1
    return result

def fu4(x):
    # min is 0 at [0 for i in range(len(x)]
    y = []
    for i in range(len(x)):
        y.append(abs(x[i]))
    return max(y)

def fu5(x):
    '''
    min is 0 at [-0.5 for i in range(len(x)]
    :param x:
    :return:
    '''
    s1 = 0
    for i in range(len(x)):
        s1 += (abs(x[i] + 0.5) ** 2)
    result = s1
    return result

def fu6(x):
    ''' Rosenbrock function
        min is 0 at [1 for i in range(len(x)]
        :param x:
        :return:
        '''
    s1 = 0
    for i in range(len(x) - 1):
        s1 += (100 * (x[i+1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2)
    result = s1
    return result

def fu7(x):
    s1 = 0
    for i in range(len(x)):
        s1 += (i * x[i] ** 4 + norm.rvs())
    result = s1
    return result

# Multimodal test functions
def fm1(x): # Eggholder Function
    '''
    min is 0 at [0 for i in range(len(x)]
    :param x: xi ∈ [-5.12, 5.12]
    :return:
    '''
    result = 0
    for i in range(len(x)):
        result += (x[i] ** 2 - 10 * math.cos(2 * math.pi * x[i]) + 10)
    return result

def fm2(x):
    '''
    GRIEWANK FUNCTION:http://www.sfu.ca/~ssurjano/griewank.html
    :param x:-600,600
    :return:
    '''
    s1 = 0
    s2 = 1
    for i in range(len(x)):
        s1 += x[i] ** 2
        s2 *= math.cos(x[i] / math.sqrt(i + 1))
    s1 = s1 / 4000
    result = s1 - s2 + 1
    return result

def fm3(x):
    # min is −418.9829 * len(x)
    # x: -500, 500
    s1 = 0
    for i in range(len(x)):
        s1 += -x[i] * math.sin(math.sqrt(abs(x[i])))
    result = s1
    return result

def fm4(x):
    '''
    ACKLEY FUNCTION:http://www.sfu.ca/~ssurjano/ackley.html
    :param x:xi ∈ [-32.768, 32.768]
    :return:
    '''
    a = 20
    b = 0.2
    c = 2 * math.pi
    s1 = 0
    s2 = 0
    for i in range(len(x)):
        s1 += x[i] ** 2
        s2 += math.cos(c * x[i])
    s1 = -a * math.exp(-b * math.sqrt(s1 / len(x)))
    s2 = -math.exp(s2 / len(x))
    result = s1 + s2 + a + math.exp(1)
    return result

def fm5(x):
    # x: -50, 50
    def y(xi):
        y = 1 + (xi + 1) / 4
        return y

    def u(xi):
        a  = 10
        k = 100
        m = 4
        if xi > a:
            u = k * pow((xi -a), m)
        elif xi < -a:
            u = k * pow((-xi - a), m)
        else:
            u = 0
        return u

    s1 = 10 * math.sin(math.pi * y(x[0])) ** 2 + (y(x[-1]) - 1) ** 2
    # s1 = 10 * math.sin(math.pi * y(x[0])) + (y(x[-1]) - 1) ** 2
    # 在已有的算法文章中使用的关于fm5的测试函数s1 = 10 * math.sin(math.pi * y(x[0])) + (y(x[-1]) - 1) ** 2
    # Xue J, Shen B. A novel swarm intelligence optimization approach: sparrow search algorithm[J].
    # Systems Science & Control Engineering, 2020, 8(1): 22-34. 中的F12
    #Mirjalili S, Lewis A. The Whale Optimization Algorithm[J]. Advances in Engineering Software,
    # 2016, 95: 51-67. 中的F12
    # 但是在这种情况下，该测试函数的理论最小值显然不是0；因此怀疑文章关于此测试函数的描述出现了问题
    # 比如当x[0]=1,x[i]=-1(i!=0)时，此时的F12<0
    for i in range(len(x)-1):
        s1 += ((y(x[i]) - 1) ** 2 * (1 + 10 * math.sin(math.pi * y(x[i+1])) ** 2))
    s2 = 0
    for i in range(len(x)):
        s2 += u(x[i])
    result = s1 * math.pi / len(x) + s2
    return result

# . Fixed-dimension test functions
# two dimension
def f21(x):
    '''
    BUKIN FUNCTION N. 6:http://www.sfu.ca/~ssurjano/bukin6.html
    :param x: x1 ∈ [-15, -5], x2 ∈ [-3, 3].
    :return:
    '''
    s1 = 100 * math.sqrt(abs(x[1] - 0.01 * x[0] ** 2))
    s2 = 0.01 * abs(x[0] + 10)
    result = s1 + s2
    return result

def f22(x):
    # min is -1.0316
    x1 = x[0]
    x2 = x[1]
    result = 4 * x1 ** 2 - 2.1 * x1 ** 4 + x1 ** 6 /3 + x1 * x2 - 4 * x2 ** 2 + 4 * x2 ** 4
    return result

def f23(x):
    # Eggholder function:
    # xi ∈ [-512, 512]
    # min is f(512, 404.2319) =-959.6407
    x1 = x[0]
    x2 = x[1]
    result = -(x2 + 47) * math.sin(math.sqrt(math.fabs(x1 + x2/2 + 47))) - x1 * math.sin(math.sqrt(math.fabs(x1 - x2 - 47)))
    return result

def f24(x):
    # Ackley's function
    # xi ∈ [-5, 5]
    # min is f(0, 0) = 0
    x1 = x[0]
    x2 = x[1]
    s1 = -20 * math.exp(-math.sqrt((x1 ** 2 + x2 ** 2)/50))
    s2 = math.exp(0.5 * math.cos(2 * math.pi * x1) + math.cos(2 * math.pi * x2))
    result = s1 - s2 + 20 + math.e
    return result

def f25(x):
    # Beale 's function
    # xi ∈ [-4.5, 4.5]
    # min is f(3, 0.5) = 0
    x1 = x[0]
    x2 = x[1]
    s1 = (1.5 - x1 + x1 * x2) ** 2
    s2 = (2.25 - x1 + x1 * x2 * x2) ** 2
    s3 = (2.625 - x1 + x1 * x2 * x2 * x2) ** 2
    result = s1 + s2 + s3
    return result

def f26(x):
    # Lévi function N.13
    # xi ∈ [-10, 10]
    # min is f(1, 1) = 0
    x1 = x[0]
    x2 = x[1]
    s1 = (math.sin(3 * math.pi * x1)) ** 2
    s2 = (x1 - 1) * (x1 - 1) * (1 + (math.sin(3 * math.pi * x2)) ** 2)
    s3 = (x2 - 1) * (x2 - 1) * (1 + (math.sin(3 * math.pi * x2)) ** 2)
    result = s1 + s2 + s3
    return result

def f27(x):
    # Cross-in-tray function:
    # xi ∈ [-10, 10]
    # min is f(±1.34941, ±1.34941) = -2.06261
    x1 = x[0]
    x2 = x[1]
    s1 = abs(100 - math.sqrt(x1 * x1 + x2 * x2) / math.pi)
    s2 = math.sin(x1) * math.sin(x2) * math.exp(s1)
    s3 = -0.0001 * math.pow((abs(s2) + 1), 0.1)
    result = s3
    return result

if __name__ == '__main__':
    pass