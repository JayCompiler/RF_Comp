# -*- coding: utf-8 -*-
"""
Created on Thu May 24 15:21:52 2018

@author: YZi
"""
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import math as mt
import gc 
#导入整个数据集以及对应的标签
gc.collect()
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
clf = RandomForestClassifier(n_estimators=1500, max_depth=None,criterion='gini',
     min_samples_split=2, random_state=None,max_features=int(mt.sqrt(dataset.shape[1])),
     n_jobs=-1)
scores = cross_val_score(clf, dataset, label,cv=2,scoring='roc_auc')
result=scores.mean() 
print(result) 
