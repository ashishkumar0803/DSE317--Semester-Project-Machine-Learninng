#!/usr/bin/env python
# coding: utf-8

# In[69]:


import csv,os,re,sys,codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib,  statistics
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression


# In[5]:


data=pd.read_csv(".\project18\exoplanet_trn_data.csv")


# In[6]:


df=data.copy()
#print(df)


# In[7]:


df1=df.isna().sum() >=0.50
print(df1)


# In[15]:


df.describe()


# In[18]:


df3.dtypes


# In[ ]:





# In[19]:


df3=df.drop(df.columns[df.isnull().sum()>0.50],axis=1)
print(df3)


# In[20]:


# Create a list of column names containing string values
string_columns = [col for col in df3.columns if df3[col].dtype == 'object']

if string_columns:
    print("Columns containing string values:", string_columns)
else:
    print("No column contains string values.")


# In[21]:


df4=df3.isna().sum()
print(df4)


# In[22]:


y=df3['pl_name'].unique()
y.size


# In[23]:


y1=df3['soltype'].unique()
y1.size


# In[24]:


y2=df3['discoverymethod'].unique()
y2.size


# In[25]:


y3=df3['pl_letter'].unique()
y3.size


# In[26]:


y4=df3['hostname'].unique()
y4.size


# In[27]:


y5=df3['pl_name'].unique()
y5.size


# In[28]:


y5=df3['disc_locale'].unique()
y5.size


# In[29]:


y6=df3['disc_facility'].unique()
y6.size


# In[30]:


y8=df3['disc_instrument'].unique()
y8.size


# In[31]:


ohe = OneHotEncoder(handle_unknown='ignore',sparse_output = False).set_output(transform = 'pandas')
ohetransform = ohe.fit_transform(df3[['soltype','discoverymethod','pl_letter','disc_locale']])


# In[32]:


ohetransform


# In[33]:


df5 = pd.concat([df3,ohetransform],axis = 1).drop(columns = ['soltype','discoverymethod','pl_letter','disc_locale'])


# In[34]:


df5.head()


# In[35]:


data_target=pd.read_csv(".\project18\exoplanet_trn_data_targets.csv",header=None)


# In[36]:


df6=data_target.drop(columns=[0])
df6


# In[37]:


df7=df6.rename(columns ={1:'target'})
df7


# In[38]:


df8 = pd.concat([df5,df7],axis = 1)
df8


# In[39]:


# Create a list of column names containing string values
string_columns = [col for col in df8.columns if df8[col].dtype == 'object']

if string_columns:
    print("Columns containing string values:", string_columns)
else:
    print("No column contains string values.")


# In[40]:


reader=df8.drop(columns=['pl_name',
                      'hostname', 'disc_pubdate', 
                      'disc_facility', 'disc_telescope', 
                      'disc_instrument', 'rv_flag', 'pul_flag', 'pl_controv_flag'],axis=1)
reader


# In[41]:


data=reader.iloc[:, :-1]
labels=reader['target']


# In[42]:


training_data, validation_data, training_cat, validation_cat = train_test_split(data, labels, 
                                                test_size=0.10, random_state=42)


# In[43]:


training_cat=[x for x in training_cat]


# In[44]:


validation_cat=[x for x in validation_cat]


# In[45]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# In[73]:


rgr = LinearRegression() 
rgr_parameters = {'rgr__positive':(True,False),}
scaler= StandardScaler()
pipeline = Pipeline([('scaler',scaler),('feature_selection',SelectKBest(mutual_info_regression ,k=20)),
                    ('rgr',rgr)])

grid = GridSearchCV(pipeline,rgr_parameters,scoring='f1_macro',cv=10,n_jobs=-1)          
grid.fit(training_data,training_cat)  
rgr= grid.best_estimator_
print(rgr)
predicted=rgr.predict(validation_data)

# Regression report
mse=mean_squared_error(validation_cat,predicted,squared=True)
print ('\n MSE:\t'+str(mse)) 
rmse=mean_squared_error(validation_cat,predicted,squared=False)
print ('\n RMSE:\t'+str(rmse))
r2=r2_score(validation_cat,predicted,multioutput='variance_weighted') 
print ('\n R2-Score:\t'+str(r2))


# In[78]:


print('### Decision Tree ### ')
rgr = DecisionTreeRegressor(random_state=40) 
rgr_parameters = {'rgr__criterion':('squared_error','friedman_mse','poisson'), 
            'rgr__max_features':('sqrt', 'log2'),
            'rgr__max_depth':(10,40,45,60),
            'rgr__ccp_alpha':(0.009,0.01,0.05,0.1),}


# In[79]:


rgr.fit(training_data,training_cat)
pipeline = Pipeline([('rgr',rgr)])

grid = GridSearchCV(pipeline,rgr_parameters,scoring='f1_macro',cv=10,n_jobs=-1)          
grid.fit(training_data,training_cat)  
rgr= grid.best_estimator_
print(rgr)
predicted=rgr.predict(validation_data)

# Regression report
mse=mean_squared_error(validation_cat,predicted,squared=True)
print ('\n MSE:\t'+str(mse)) 
rmse=mean_squared_error(validation_cat,predicted,squared=False)
print ('\n RMSE:\t'+str(rmse))
r2=r2_score(validation_cat,predicted,multioutput='variance_weighted') 
print ('\n R2-Score:\t'+str(r2))


# In[77]:


print('### Random Forest ###')
rgr = RandomForestRegressor(max_features=None)
rgr_parameters = {
            'rgr__criterion':('squared_error','friedman_mse','poisson'),       
            'rgr__n_estimators':(30,50,100),
            'rgr__max_depth':(10,20,30),
            'rgr__max_features':( 'sqrt', 'log2'),
            } 



rgr.fit(training_data,training_cat)
pipeline = Pipeline([('rgr',rgr)])

grid = GridSearchCV(pipeline,rgr_parameters,scoring='f1_macro',cv=10,n_jobs=-1)          
grid.fit(training_data,training_cat)  
rgr= grid.best_estimator_
print(rgr)
predicted=rgr.predict(validation_data)

# Regression report
mse=mean_squared_error(validation_cat,predicted,squared=True)
print ('\n MSE:\t'+str(mse)) 
rmse=mean_squared_error(validation_cat,predicted,squared=False)
print ('\n RMSE:\t'+str(rmse))
r2=r2_score(validation_cat,predicted,multioutput='variance_weighted') 
print ('\n R2-Score:\t'+str(r2))


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor


# In[80]:


rgr = KNeighborsRegressor()
rgr_parameters = { 'rgr__weights':('uniform', 'distance'),
                 'rgr__algorithm':('ball_tree', 'kd_tree', 'brute'),
                  'rgr__n_neighbors':(3,5,7,11,13)
                 }

rgr.fit(training_data,training_cat)
pipeline = Pipeline([('rgr',rgr)])

grid = GridSearchCV(pipeline,rgr_parameters,scoring='f1_macro',cv=10,n_jobs=-1)          
grid.fit(training_data,training_cat)  
rgr= grid.best_estimator_
print(rgr)
predicted=rgr.predict(validation_data)

# Regression report
mse=mean_squared_error(validation_cat,predicted,squared=True)
print ('\n MSE:\t'+str(mse)) 
rmse=mean_squared_error(validation_cat,predicted,squared=False)
print ('\n RMSE:\t'+str(rmse))
r2=r2_score(validation_cat,predicted,multioutput='variance_weighted') 
print ('\n R2-Score:\t'+str(r2))     


# In[82]:


print('### Ridge Regression ###')
training_data, validation_data, training_cat, validation_cat = train_test_split(data, labels, 
                                                test_size=0.10, random_state=42)
training_cat=training_cat.astype('int')
pipeline = Pipeline([('rgr', rgr),])
rgr = Ridge(alpha=1.0) 
rgr_parameters = {'rgr__solver':('auto','lbfgs',)}

rgr.fit(training_data,training_cat)

grid = GridSearchCV(pipeline,rgr_parameters,scoring='f1_macro',cv=10,n_jobs=-1)          
grid.fit(training_data,training_cat)  
rgr= grid.best_estimator_
print(rgr)
predicted=rgr.predict(validation_data)

# Regression report
mse=mean_squared_error(validation_cat,predicted,squared=True)
print ('\n MSE:\t'+str(mse)) 
rmse=mean_squared_error(validation_cat,predicted,squared=False)
print ('\n RMSE:\t'+str(rmse))
r2=r2_score(validation_cat,predicted,multioutput='variance_weighted') 
print ('\n R2-Score:\t'+str(r2))


# In[91]:


print('\n\t### SVM Regressor ### \n')
rgr = SVR(epsilon=0.2)  
rgr_parameters = {
            'rgr__C':(0.1,1,100),
            'rgr__kernel':('linear','rbf','poly','sigmoid'),
            }
            
            
rgr.fit(training_data,training_cat)
pipeline = Pipeline([('scaler',scaler),('feature_selection',SelectKBest(mutual_info_regression ,k=4)),
                    ('rgr',rgr)])

grid = GridSearchCV(pipeline,rgr_parameters,scoring='f1_macro',cv=10,n_jobs=-1)          
grid.fit(training_data,training_cat)  
rgr= grid.best_estimator_
print(rgr)
predicted=rgr.predict(validation_data)

# Regression report
mse=mean_squared_error(validation_cat,predicted,squared=True)
print ('\n MSE:\t'+str(mse)) 
rmse=mean_squared_error(validation_cat,predicted,squared=False)
print ('\n RMSE:\t'+str(rmse))
r2=r2_score(validation_cat,predicted,multioutput='variance_weighted') 
print ('\n R2-Score:\t'+str(r2))       


# In[90]:


print('\n\t### AdaBoost Regression ### \n')
be1 = LinearRegression()              
be2 = DecisionTreeRegressor(random_state=0)
be3 = Ridge(alpha=1.0,solver='lbfgs',positive=True)             
rgr = AdaBoostRegressor(n_estimators=100)
pipeline = Pipeline([('rgr', rgr),])
rgr_parameters = {
            'rgr__base_estimator':(be1,be2,be3),
            'rgr__random_state':(0,10),}

rgr.fit(training_data,training_cat)

grid = GridSearchCV(pipeline,rgr_parameters,scoring='f1_macro',cv=10,n_jobs=-1)          
grid.fit(training_data,training_cat)  
rgr= grid.best_estimator_
print(rgr)
predicted=rgr.predict(validation_data)

# Regression report
mse=mean_squared_error(validation_cat,predicted,squared=True)
print ('\n MSE:\t'+str(mse)) 
rmse=mean_squared_error(validation_cat,predicted,squared=False)
print ('\n RMSE:\t'+str(rmse))
r2=r2_score(validation_cat,predicted,multioutput='variance_weighted') 
print ('\n R2-Score:\t'+str(r2))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




