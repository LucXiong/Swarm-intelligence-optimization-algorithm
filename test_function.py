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

# Unimodal test functions
def f1(x):
    # min is 0 at [0 for i in range(len(x)]
    s1 = 0
    for i in range(len(x)):
        s1 += x[i] ** 2
    return s1

def f2(x):
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

def f3(x):
    # min is 0 at [0 for i in range(len(x)]
    s1 = 0
    for i in range(len(x)):
        s2 = 0
        for j in range(i):
            s2 += abs(x[i])
        s1 += s2 ** 2
    result = s1
    return result

def f4(x):
    # min is 0 at [0 for i in range(len(x)]
    y = []
    for i in range(len(x)):
        y.append(abs(x[i]))
    return max(y)

def f5(x):
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

# Multimodal test functions
def f6(x): # Eggholder Function
    '''
    min is 0 at [0 for i in range(len(x)]
    :param x: xi ∈ [-5.12, 5.12]
    :return:
    '''
    result = 0
    for i in range(len(x)):
        result += (x[i] ** 2 - 10 * math.cos(2 * math.pi * x[i]) + 10)
    return result

def f7(x):
    '''
    GRIEWANK FUNCTION:http://www.sfu.ca/~ssurjano/griewank.html
    :param x:
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

def f8(x):
    # min is −418.9829 * len(x)
    s1 = 0
    for i in range(len(x)):
        s1 += -x[i] * math.sin(math.sqrt(abs(x[i])))
    result = s1
    return result

def f9(x):
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

# Fixed-dimension test functions
def f10(x):
    '''
    BUKIN FUNCTION N. 6:http://www.sfu.ca/~ssurjano/bukin6.html
    :param x:xi ∈ [-32.768, 32.768]
    :return:
    '''
    s1 = 100 * math.sqrt(abs(x[1] - 0.01 * x[0] ** 2))
    s2 = 0.01 * abs(x[0] + 10)
    result = s1 + s2
    return result

def f11(x):
    # min is -1.0316
    x1 = x[0]
    x2 = x[1]
    result = 4 * x1 ** 2 - 2.1 * x1 ** 4 + x1 ** 6 /3 + x1 * x2 - 4 * x2 ** 2 + 4 * x2 ** 4
    return result

def f12(x):
    x1 = x[0]
    x2 = x[1]
    result = -(x2 + 47) * math.sin(math.sqrt(math.fabs(x1 + x2/2 + 47))) - x1 * math.sin(math.sqrt(math.fabs(x1 - x2 - 47)))
    return result


