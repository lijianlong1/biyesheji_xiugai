# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:38:13 2020

@author: 龙
"""
from bert_serving.client import BertClient
from termcolor import colored
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.spatial.distance import pdist

a=[-1,2,3,4]
b=[1,2,3,4]
c=np.array(a)
d=np.array(b)
print(np.dot(c,d))
print(VectorCosine(c,d))
def cos_sim(x,y):
    #输入的x，y应为一个np数组
    cos=np.dot(x,y)/(math.sqrt(np.dot(x,x))*math.sqrt(np.dot(y,y)))
    return cos
print(cos_sim(c,d))