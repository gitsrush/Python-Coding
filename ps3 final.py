#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from cleands import *
import matplotlib.pyplot as plt


# In[3]:


def bias_sim(n,rho,a):
    #print('For n=',n, 'rho=',rho, 'alpha=',a)
    #data generation
    np.random.seed(110)
    e=np.random.normal(size=(n+100,1))
    y=np.zeros(shape=(n+100,1))
    for i in range(1,n+100):
        y[i] += rho*y[i-1]+e[i]+a
    y = y[100:]
    xmat = y[:-1]
    ymat = y[1:]
    df=pd.DataFrame(data=np.column_stack((xmat,ymat)),columns=['X','Y'])
    #print(df)
    
    #predict using OLS regression
    model = LeastSquaresRegressor(*add_intercept(['X'],'Y',df))
    #print('Coefficients after OLS:\n',model.tidy)
    bias1 = model.params[1]-rho
    #print('Bias=',bias1)
    bias=[bias1]*(n-1)
    sigma=model.std_error[1]
    sigma1=sigma**2
    sigma2=[sigma1]*(n-1)
    alpha1=model.params[0]
    alpha=[alpha1]*(n-1)
    u=model.residuals.tolist()
    t1=1/(n-1)
    t=[t1]*(n-1)
    rho1=[rho]*(n-1)
    df2=pd.DataFrame(data=np.column_stack((rho1,t,alpha,sigma2,u,bias)),columns=['rho','1/T','alpha','sigma2','ut','Predicted_bias'])
    simulation = LeastSquaresRegressor(*add_intercept(['rho','1/T','alpha','sigma2','ut'],'Predicted_bias',df2))
    #print('\nEstimated simulation Coefficients:\n',simulation.tidy)
    #print('Predicted bias from simulation:\n',simulation.predict(df2))
    return simulation.params, bias1, sigma1


n=[10,20,30,40,50,60,70,80,90,100]
Rho=[0.1,0.2,0.3,0.4,0.55,0.6,0.7,0.8,0.9,1.1]
a=[1]
b=[]
variance=[]
for i in Rho:
    bias_list = []
    for k in n:
        ax = plt.subplot(111)
        result1,result2,result3 = bias_sim(k,i,a[0])
        b+=[result1]
        bias_list+=[result2]
        variance+=[result3]
    ax.plot(n, bias_list, label= 'rho = {}'.format(i))
plt.title('Bias Vs Sample Size')
plt.xlabel('Sample Size')
plt.ylabel('Bias')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
plt.show()

for j in range(0,10):
    print('Summary:\n For sample size={0}, rho={1}, alpha={2}, sigma_sq={3}\n bias={4}\n Beta-coefficients from simulation={4}'.format(n[j],Rho[j],a[0],variance[j],bias_list[j],b[j]))


# In[ ]:




