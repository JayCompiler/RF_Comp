# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 17:02:06 2018
逻辑分类器
@author: zhang_yu
"""
#from sklearn.model_selection import cross_val_score
import numpy as np
import math as mt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import time
import gc 
gc.collect()
start = time.clock()
#导入整个数据集以及对应的标签
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
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred_class = lr.predict(X_test)
# calculate accuracy
print ("acc:",metrics.accuracy_score(y_test, y_pred_class))
#计算空准确率
print("null acc:",max(y_test.mean(), 1-y_test.mean()))
# 混淆矩阵
print ("混淆矩阵：",metrics.confusion_matrix(y_test, y_pred_class))
#scores = cross_val_score(lr, dataset, label,cv=2,scoring='roc_auc')
#result=scores.mean() 
#print(result) 
elapsed = (time.clock() - start)
print("Time used:",elapsed)
