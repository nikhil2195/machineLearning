#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

df=pd.read_csv("Admission_Predict.csv")
df=df.drop("Serial No.",axis=1)
df.head()


# In[17]:


X=df.drop("Chance of Admit",axis=1)
y=df["Chance of Admit"]


# In[18]:


X=X.values
y=y.values


# In[32]:


X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2)
clf=LinearRegression()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# In[33]:


with open('model','wb') as f:
    pickle.dump('model',f)


# In[ ]:




