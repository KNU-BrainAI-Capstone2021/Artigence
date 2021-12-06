# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 17:17:39 2021

@author: PC00
"""
import numpy as np
decision = [0,1,2,1,2,1,1,2,0,1,2,0,2,1,2,0,2,1,2,0]
decision_index = []
count_list = []
last_index = []
count_list.append(decision.count(0))
decision_index.append( np.where(np.array(decision)==0)[0] )

count_list.append(decision.count(1))
decision_index.append( np.where(np.array(decision)==1)[0] )

count_list.append(decision.count(2))
decision_index.append( np.where(np.array(decision)==2)[0] )

most_result = np.where(np.array(count_list)==max(count_list))[0]

for i in most_result:
    last_index.append( decision_index[i][-1] )
    

decision_result = most_result[last_index.index(max(last_index))]

print(decision_result)