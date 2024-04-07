#!/usr/bin/env python
# coding: utf-8

# # CAR PRICE PREDICTION WITH MACHINE LEARNING:
# (Task-3)

# In[1]:


#importing basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model


# In[2]:


data_set=pd.read_csv(r"C:\Users\HP\OneDrive\Documents\oasis infobytes\car data.csv")
data_set.head(50)


# In[3]:


data_set.shape


# In[4]:


data_set.isnull().sum() #checking the null value


# In[5]:


data_set.info()


# In[6]:


data_set.describe()


# In[7]:


data_set.columns


# In[ ]:





# In[ ]:





# # Data Modifications

# In[8]:


inputs=data_set.drop(['Car_Name','Owner','Selling_type'],axis='columns')
inputs


# In[9]:


target=data_set.Selling_Price
target


# In[ ]:





# In[ ]:





# # Data visualization

# In[10]:


plt.figure(figsize=(5,4))
sns.countplot(x='Fuel_Type',data=data_set)


# In[11]:


plt.figure(figsize=(5,4))
sns.countplot(x='Transmission',data=data_set)


# In[12]:


plt.figure(figsize=(5,4))
sns.countplot(x='Selling_type',data=data_set)


# In[13]:


plt.figure(figsize=(5,5))
sns.kdeplot(data_set['Selling_Price'])


# In[14]:


plt.figure(figsize=(5,5))
sns.kdeplot(data_set['Present_Price'])


# In[15]:


z=data_set.drop(['Car_Name', 'Year', 'Driven_kms',
       'Fuel_Type', 'Selling_type', 'Transmission', 'Owner'],
      axis=1)
z


# In[16]:


sns.kdeplot(z)


# In[17]:


sns.heatmap(data_set.corr(),cmap='Blues')


# In[ ]:





# In[ ]:





# # Training the model

# In[18]:


#Encoding
from sklearn.preprocessing import LabelEncoder
Numerics=LabelEncoder()


# In[22]:


inputs['Fuel_Type']=Numerics.fit_transform(inputs['Fuel_Type'])    #encoded fuel type CNG-0 , Diesel-1, Petrol-2
inputs['Transmission']=Numerics.fit_transform(inputs['Transmission']) #encoded transmission type  Automatic-0 ,Manual-1
inputs


# In[23]:


model=linear_model.LinearRegression()
inputs=inputs.values 


# In[24]:


model.fit(inputs,target)


# In[ ]:





# In[ ]:





# # Final result of prediction

# In[31]:


prediction=model.predict( [[2020,1500,100,1000,2,0]])   # (Year,Selling_Price,Present_Price,Driven_kms,Fuel_Type,Transission)
print("Car price predicted value:",prediction)              #encoded fuel type CNG-0 , Diesel-1, Petrol-2
                                                            #encoded transmission type  Automatic-0 ,Manual-1


# In[ ]:





# In[ ]:


#Thanking You...


#                                                                                                  - Thiruvalluvan G
