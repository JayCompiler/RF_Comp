# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:49:12 2018

@author: JayMining
"""
#from sklearn.ensemble import RandomForestClassifier
#X = [[0, 0], [1, 1]]
#Y = [0, 1]
#clf = RandomForestClassifier(n_estimators=10)
#clf = clf.fit(X, Y)
import numpy as np

a=np.array([[1,2],[3,4],[5,6],[7,8]])
b=np.array([[-1,-2],[-3,-4],[-5,-6],[-7,-8]])
c=np.append(a,b)
print(c)
print(c.shape)