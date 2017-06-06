# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:54:56 2017

@author: rogue
"""

#computes the sum of squared elements
def sum_of_squares(v):
	return sum(v_i ** 2 for v_i in v)
	
#定义一元函数梯度计算
def difference_quotient(f,x,h):
	return (f(x+h)-f(x))/h

#定义多元函数梯度计算
#定义偏导估计函数，compute the i-th partial difference quotient of f at v
def partial_difference_quotient(f,v,i,h):
	w = [v_j + (h if j==i else 0) for j, v_j in enumerate(v)]
	return (f(w)-f(v))/h

#再定义梯度估计函数
def estimate_gradient(f,v,h=0.00001):
	return [partial_difference_quotient(f,v,i,h) for i,_ in enumerate(v)]