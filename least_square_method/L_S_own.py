# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:21:58 2017

@author: rogue
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:36:26 2017

@author: rogue
"""

import numpy as np
from scipy import optimize
import random
from matplotlib import pyplot as plt

random.seed()

x = [x_i*random.randrange(8,12)/10 for x_i in range(100)]
y = [y_i*random.randrange(8,12)/10 for y_i in range(200,300)]
def residuals(p): 
    "计算以p为参数的直线和原始数据之间的误差"
    k, b = p
    return y - (k*x + b)

# leastsq使得residuals()的输出数组的平方和最小，参数的初始值为[1,0]
r = optimize.leastsq(residuals, [1, 1]) 
k, b = r[0]
print ("k =",k, "b =",b)

fig, ax = plt.subplots(1,1)

plt.plot(x, y, "*r")
X0 = np.linspace(0, 100, 600)
Y0 = k*X0 + b
plt.plot(X0, Y0)