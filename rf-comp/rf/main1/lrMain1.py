# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 17:02:06 2018
逻辑分类器
@author: zhang_yu
"""
from sklearn.model_selection import cross_val_score
import numpy as np
import math as mt
from sklearn.linear_model import LogisticRegression
import time
import gc 
gc.collect()
start = time.clock()
# 导入整个数据集以及对应的标签
dataset1=np.loadtxt('dataset1k2')
dataset2=np.loadtxt('dataset1k3')
dataset3=np.loadtxt('dataset1k4')
dataset4=np.loadtxt('dataset1k5')
dataset5=np.loadtxt('dataset1k6')
dataset6=np.loadtxt('dataset1k7')
dataset=np.concatenate((dataset1,dataset2,dataset3,dataset4,dataset5,dataset6),axis=1)
print(dataset1.shape)

#np.savetxt('test',dataset)
label=np.loadtxt('label1')
#查询序列
querySequence=dataset[39]
#待查询序列
dataset=np.array(dataset[0:39])
dataset=dataset-querySequence
np.transpose(label)
print(label)
print(mt.sqrt(dataset.shape[1]))
lr = LogisticRegression(C=1000.0, random_state=0)
scores = cross_val_score(lr, dataset, label,cv=2,scoring='roc_auc')
result=scores.mean() 
print(result) 