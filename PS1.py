#!/usr/bin/env python
# coding: utf-8

# In[1]:


#unique holding patterns for different values of a and b

for a in range(1,11):
    for b in range(1,11):
        my_dict={}
        key=1
        for x in range(1,1001):
            init=x
            ls=[]
            count=0
            while x not in ls:
                ls+=[x]
                x=x//2 if x%2==0 else a*x+b
                count+=1
                if x in ls:
                    index=ls.index(x)
                    pattern=list(ls[index:])
                    pattern.sort()
                    if pattern not in my_dict.values():
                        my_dict[key]=pattern
                        key+=1
                        break
                if count==100:
                    break
        print('a=',a, ' b=',b, '  unique_holding_patterns=',len(my_dict))

