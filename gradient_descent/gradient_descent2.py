# -*- coding: utf-8 -*-
"""
Created on Tue May  2 08:47:39 2017

@author: rogue
"""

#梯度下降算法

##定义步长
def step(v, direction, step_size):
	return [v_i + step_size * direction_i for v_i ,direction_i in zip (v, direction)]

##定义梯度
def sum_of_squares_gradient(v):
	return [2 * v_i for v_i in v]

##定义向量的距离
######################
#减法定义
def vector_substract(v,w):
	return [v_i - w_i for v_i, w_i in zip(v,w)]
									
#乘法定义
def dot(v,w):
	return sum(v_i * w_i for v_i, w_i in zip(v,w))
												
#向量的平方和定义
def sum_of_squares(v):
	return dot(v,v)

#向量的距离
def distance(v,w):
	return sum_of_squares(vector_substract(v,w))

##进行迭代计算找出最低点
#随机选择起点
import random
v = [random.randint(-10, 10) for i in range(3)]

#设置阈值
tolerance = 0.0001
max_iter = 100000
iter = 1

while True:
	gradient = sum_of_squares_gradient(v)
	next_v = step(v, gradient, 0.001)
	if (distance(next_v, v) < tolerance) or (iter > max_iter):
		break
	v = next_v
	iter += 1
print(v, iter)
												