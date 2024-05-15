#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing libraries
import numpy as np
import pandas as pd


# In[4]:


#get california data
from sklearn.datasets import fetch_california_housing
housing=fetch_california_housing()
housing


# In[5]:


#converting data and giving column names
housing_df= pd.DataFrame(housing["data"],columns=housing['feature_names'])
housing_df


# In[6]:


#setting target value
housing_df['MedHouseVal']=housing['target']
#renaming target value
housing_df.rename(columns={'MedHouseVal':'target'},inplace=True)


# In[7]:


housing_df


# In[10]:


#ignoring warnings
import warnings
warnings.filterwarnings('ignore')


# In[11]:


from sklearn.model_selection import train_test_split
#importing algorithms
from sklearn.linear_model import Ridge
#splitting data into features and targets
np.random.seed(42)

#create the data
x=housing_df.drop('target',axis=1)
y=housing_df['target']

#split into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#instantiate and fit the model
model=Ridge()
model.fit(x_train,y_train)

#check the score of model
model.score(x_test,y_test)


# In[12]:


#imporving our model
#importing model
from sklearn.ensemble import RandomForestRegressor

#setup random seed
np.random.seed(42)

#create data
x=housing_df.drop('target',axis=1)
y=housing_df['target']

#splitting data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#creating model
model=RandomForestRegressor()
model.fit(x_train,y_train)
model.score(x_test,y_test)


# In[14]:


y_preds=model.predict(x_test)
y_preds


# In[28]:


y_test


# In[27]:


predicted_value


# In[29]:


#comparing prediction to truth values
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,y_preds)


# #### from above we find that the mean_absolute_error value is 0.3266

# In[31]:


model.score(x_test,y_test)


# #### model's score is 0.8065

# ### We have Trained Our model with 80% accuracy

# In[ ]:




