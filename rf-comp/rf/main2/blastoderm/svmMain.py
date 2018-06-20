# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 16:47:10 2018
svm 分类
@author: zhang_yu
"""

from sklearn.model_selection import cross_val_score
import numpy as np
import math as mt
from sklearn import svm
import time

start = time.clock()
#导入整个数据集以及对应的标签
posdataset=np.loadtxt('posk6')
negdataset=np.loadtxt('negk6')
size=posdataset.shape[0]
count=0
#处理数据集
m=int(size*(size-1))
n=posdataset.shape[1]
dataset=np.zeros((m,n))
label1=np.ones((int(m/2)))
label2=np.zeros((int(m/2)))
label=np.concatenate((label1,label2))
for i in range(0,len(posdataset)):
    for j in range(i+1,len(posdataset)):
        dataset[count]=np.abs(posdataset[i]-posdataset[j])
        count=count+1
for i in range(0,len(negdataset)):
    for j in range(i+1,len(negdataset)):
        dataset[count]=np.abs(negdataset[i]-negdataset[j])
        count=count+1    
#np.transpose(label)
print(mt.sqrt(dataset.shape[1]))
clf= svm.SVC()
scores = cross_val_score(clf, dataset, label,cv=5,scoring='roc_auc')
result=scores.mean() 
print(result) 
elapsed = (time.clock() - start)
print("Time used:",elapsed)