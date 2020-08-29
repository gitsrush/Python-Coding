#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[74]:


#data-generation
np.random.seed(100)
npx = np.random.randn(100, 10)
npy = np.random.randn(100)


# In[75]:


#ridge-regression using closed form solution, here beta is estimated by minimizing both SSE + penalty term
def ridge_regression(lamda,x,y):
    n,r = x.shape
    i = np.eye(r)
    l_mat = lamda*i
    b = np.linalg.solve(x.T@x+l_mat,x.T@y)
    e = y-x@b
    se = e**2
    mse = se.mean()
    return b, mse


# In[82]:


mse = []
params = []
lamda = np.linspace(0,10000,1000)
for i in range(0,len(lamda)):
    r1,r2=ridge_regression(lamda[i],npx,npy)
    params += [r1]
    mse += [r2]


# In[83]:


plt.figure(figsize=(20, 6))

plt.subplot(121)
ax = plt.gca()
ax.plot(lamda, params)
ax.set_xscale('log')
plt.xlabel('lambda')
plt.ylabel('Beta co-efficients')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')

plt.subplot(122)
ax = plt.gca()
ax.plot(lamda, mse)
ax.set_xscale('log')
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.title('Coefficient error as a function of the regularization')
plt.axis('tight')

plt.show()


# In[ ]:




