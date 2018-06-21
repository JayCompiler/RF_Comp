# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:56:25 2018
RandomForest
@author: zhang_yu
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import math as mt
from sklearn import metrics
import gc 
#导入整个数据集以及对应的标签
gc.collect()
posdataset=np.loadtxt('posk6')
negdataset=np.loadtxt('negk6')
size=posdataset.shape[0]
count=0
#处理数据集
avgSize=int(size*(size-1)/2)

poSize=int(size*(size-1)/40)
print("正例大小：",poSize)
neSize=int(size*(size-1)/5)
print("负例大小：",neSize)

m=poSize+neSize
n=posdataset.shape[1]


posTol=np.zeros((avgSize,n))
negTol=np.zeros((avgSize,n))


label1=np.ones(poSize)
label2=np.zeros(neSize)

label=np.concatenate((label1,label2))

for i in range(0,len(posdataset)):
    for j in range(i+1,len(posdataset)):
        posTol[count]=np.abs(posdataset[i]-posdataset[j])
        count=count+1
       
count=0
for i in range(0,len(negdataset)):
    for j in range(i+1,len(negdataset)):
        negTol[count]=np.abs(negdataset[i]-negdataset[j])
        count=count+1   
#打乱，构造数据集
np.random.shuffle(posTol)  
np.random.shuffle(negTol)
pos=np.array(posTol[0:poSize])
neg=np.array(posTol[0:neSize])
dataset=np.concatenate((pos,neg),axis=0)

#split the dataset and label:
X_train, X_test, y_train, y_test = train_test_split(dataset, label, random_state=0)

#np.transpose(label)
print(mt.sqrt(dataset.shape[1]))
rf = RandomForestClassifier(n_estimators=1500, max_depth=None,criterion='gini',
     min_samples_split=2, random_state=None,max_features=int(mt.sqrt(dataset.shape[1])),
     n_jobs=-1)
rf.fit(X_train,y_train)
y_pred_class = rf.predict(X_test)
# calculate accuracy
print ("acc:",metrics.accuracy_score(y_test, y_pred_class))
#计算空准确率
print("null acc:",max(y_test.mean(), 1-y_test.mean()))
# 混淆矩阵
print ("混淆矩阵：",metrics.confusion_matrix(y_test, y_pred_class))
#clf = RandomForestClassifier(n_estimators=1500, max_depth=None,criterion='gini',
#     min_samples_split=2, random_state=None,max_features=int(mt.sqrt(dataset.shape[1])),
#     n_jobs=-1)
#scores = cross_val_score(clf, dataset, label,cv=10,scoring='roc_auc')
#result=scores.mean() 
#print(result) 