# ！usr/bin/env python
# -*- coding: utf-8 -*-
# Time : 2022/10/18 13:33
# @Author : LucXiong
# @Project : Model
# @File : test_function_20221018.py

'''
This file contains test functions in Xue's paper[1], including unimodal test functions(F1~F7), multimodal test functions(F8~F11) and fixed-dimension test functions(F13~F16).
Multimodal test function(F12 is the function fm5 in the test_function.py file) may have errors and it's not included in this file.
Fixed-test functions(F17~F19) are not included in this file, as the coefficient in F17~F19 isn't found.

There are seven unimodal test functions, four multimodal test functions and four fixed-test functions.
Test functions in this file may appear in the file(test_function,py), too.

[1] Xue Jiankai, Shen Bo. A novel swarm intelligence optimization approach: sparrow search algorithm[J]. Systems Science & Control Engineering, 2020, 8(1): 22-34.
'''
import math
from scipy.stats import norm

# Unimodal test functions
def F1(x):
    # xi ∈ [-100, 100] and min is 0
    s1 = 0
    for i in range(len(x)):
        s1 += (x[i] ** 2)
    return s1

def F2(x):
    # xi ∈ [-10, 10] and min is 0
    s1 = 0
    s2 = 1
    for i in range(len(x)):
        s1 += abs(x[i])
        s2 *= abs(x[i])
    result = s1 + s2
    return result

def F3(x):
    # xi ∈ [-100, 100] and min is 0
    s1 = 0
    for i in range(len(x)):
        s2 = 0
        for j in range(i):
            s2 += abs(x[i])
        s1 += s2 ** 2
    return s1

def F4(x):
    # xi ∈ [-100, 100] and min is 0
    y = []
    for i in range(len(x)):
        y.append(abs(x[i]))
    return max(y)

def F5(x):
    # xi ∈ [-30, 30] and min is 0
    s1 = 0
    for i in range(len(x) - 1):
        s1 += (100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2)
    return s1

def F6(x):
    # xi ∈ [-100, 100] and min is 0
    s1 = 0
    for i in range(len(x)):
        s1 += ((x[i] + 0.5) ** 2)
    return s1

def F7(x):
    # xi ∈ [-1.28, 1.28] and min is 0
    s1 = 0
    for i in range(len(x)):
        s1 += (i * x[i] ** 4 + norm.rvs())
    return s1

# Multimodal test functions
def F8(x):
    # xi ∈ [-500, 500] and min is -418.9829 * len(x)
    s1 = 0
    for i in range(len(x)):
        s1 += (-x[i] * math.sin(math.sqrt(abs(x[i]))))
    return s1

def F9(x):
    # xi ∈ [-5.12, 5.12] and min is 0
    result = 0
    for i in range(len(x)):
        result += (x[i] ** 2 - 10 * math.cos(2 * math.pi * x[i]) + 10)
    return result

def F10(x):
    # xi ∈ [-32.768, 32.768] and min is 0
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

def F11(x):
    # xi ∈ [-600, 600] and min is 0
    s1 = 0
    s2 = 1
    for i in range(len(x)):
        s1 += x[i] ** 2
        s2 *= math.cos(x[i] / math.sqrt(i + 1))
    s1 = s1 / 4000
    result = s1 - s2 + 1
    return result

# Fixed-dimension test functions # two dimension
def F13(x):
    # x[0] and x[1] ∈ [-5, 5] and min is -1.0316
    x1 = x[0]
    x2 = x[1]
    result = 4 * x1 ** 2 - 2.1 * x1 ** 4 + x1 ** 6 / 3 + x1 * x2 - 4 * x2 ** 2 + 4 * x2 ** 4
    return result

def F14(x):
    # x[0] and x[1] ∈ [0, 14] and min is 0
    x1 = x[0]
    x2 = x[1]
    s1 = math.sin(math.pi * (x1-2)) * math.sin(math.pi * (x2-2)) / (math.pi * (x2-2) * math.pi * (x1-2))
    s2 = 2 + (x1-7) ** 2 + (x2-7) ** 2
    result = (1 - pow(abs(s1), 5)) * s2
    return result

def F15(x):
    # x[0] and x[1] ∈ [-10, 10] and min is -1
    x1 = x[0]
    x2 = x[1]
    s1 = abs(100 - math.sqrt(x1 * x1 + x2 * x2) / math.pi)
    s2 = math.sin(x1) * math.sin(x2) * math.exp(s1)
    s3 = -math.pow((abs(s2) + 1), -0.1)
    return s3
def F16(x):
    # x[0] and x[1] ∈ [-20, 20] and min is -1
    x1 = x[0]
    x2 = x[1]
    beta = 15
    m = 5
    s1 = math.exp(-pow(x1/beta, 2*m)-pow(x2/beta, 2*m))
    s2 = 2*math.exp(-pow(x1, 2)-pow(x2, 2))
    s3 = math.cos(x1)*math.cos(x2)
    result = (s1 - s2) * s3 * s3
    return result



