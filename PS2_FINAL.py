#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


class linear_model:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.b = np.linalg.solve(x.T@x,x.T@y)
        e = y-x@self.b
        self.vb = self.vcov_b(e)
        self.se = np.sqrt(np.diagonal(self.vb))
        self.tstat = self.b/self.se
    def vcov_b(self,e):
        x = self.x
        return e.var()*np.linalg.inv(x.T@x)

class white(linear_model):
    def vcov_b(self,e):
        x = self.x
        meat = np.diagflat(e.values**2)
        bread = np.linalg.inv(x.T@x)@x.T
        sandwich = bread@meat@bread.T
        return sandwich

class newey_west(linear_model):
    def vcov_b(self,e):
        x = self.x
        t=len(e)
        sum=0
        for i in range(1,t):
            sum+=(e.values[i]*e.values[i-1])
        nwcov=sum/t
        upper = np.zeros((t,t))
        upper[np.triu_indices(t,0)] = nwcov
        upper[np.triu_indices(t,2)] = 0
        ut = upper.T
        eye = np.eye(upper.shape[0])
        new = (upper+ut)-eye*np.diagonal(upper)
        np.fill_diagonal(new,e.var())
        meat = new
        bread = np.linalg.inv(x.T@x)@x.T
        sandwich = bread@meat@bread.T
        return sandwich


# In[3]:


df = pd.read_csv('BWGHT.csv')
df['(intercept)'] = 1
x = df[['(intercept)','cigs','faminc']]
y = df['bwght']
newey_west(x,y).vb

