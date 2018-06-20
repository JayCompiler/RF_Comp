# -*- coding: utf-8 -*-
"""
Created on Sun May 20 14:42:12 2018
@author: YZi
"""


from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import math as mt
# 导入整个数据集以及对应的标签
dataset=np.loadtxt('dataset1k2')
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
clf = RandomForestClassifier(n_estimators=1000, max_depth=None,criterion='gini',
     min_samples_split=2, random_state=None,max_features=int(mt.sqrt(dataset.shape[1])),
     n_jobs=-1)
scores = cross_val_score(clf, dataset, label,cv=5,scoring='roc_auc')
result=scores.mean() 
print(result) 


                 
#正序列
#pos=np.zeros((20,dataset.shape[1]))
#负序列
#neg=np.zeros((19,dataset.shape[1]))
#pC=0;
#nC=0;
#count=0
#print(pos.shape)
#根据标签划分为正案例集，负案例集
#for i in label:
#    if(i==1.0):
#        pos[pC]=dataset[count]
#       # print("正数"+str(pC)+":"+str(count))
#        pC=pC+1
#        count=count+1
#    else:
#        neg[pC]=dataset[count]
#      #  print("负数",str(nC),":",str(count))
#        nC=nC+1
#        count=count+1
#pos=pos-querySequence
#neg=neg-querySequence



