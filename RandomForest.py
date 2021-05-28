#!/usr/bin/env python
# coding: utf-8

# ### Import Packages

# In[16]:


import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[17]:


## Concatenate csv

path = 'processed_data' 
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

dataframe = pd.concat(li, axis=0, ignore_index=True)
## remove rows with Nan values.
dataframe.dropna()
## Save the dataframe
dataframe.to_csv('dataframe.csv',index=False)


# In[18]:


df=pd.read_csv('dataframe.csv')


# In[20]:


## Filtering the data frame to reduce its size

Z=df.primary_label.value_counts()[0:25]
l=Z.index
#l
for i in range(0,25):    
    if i==0:
        filter_df=df[df['primary_label']==l[i]]
    else:
        filter_df=filter_df.append(df[df['primary_label']==l[i]],ignore_index=True)
DF=filter_df.dropna()
#DF.info()  


# In[22]:


#DF.columns


# In[23]:


# Features columns and target column

X=DF[['peak0_accuracy', 'peak0_freqency', 'peak0_prominence','peak0_width', 'peak0_width_height',
      'peak1_accuracy', 'peak1_freqency', 'peak1_prominence', 'peak1_width', 'peak1_width_height',
       'peak2_accuracy', 'peak2_freqency', 'peak2_prominence', 'peak2_width','peak2_width_height']]
Y=DF['primary_label']


# In[24]:


#X.info(memory_usage='deep')
#len(X)


# In[25]:


#print(Y)


# In[26]:


y = Y.astype("category").cat.codes


# In[27]:


#y


# In[28]:


#Train_split for original data
X0_train, X0_test, y0_train, y0_test = train_test_split(X,y,test_size = 0.3, stratify=y)


# In[13]:


#Normalize the data and PCA I

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipe = Pipeline([('scale',StandardScaler()),
                     ('pca',PCA(n_components =10))])
pipe.fit(X)
fit=pipe.transform(X)
## train-split after PCA I
X_train, X_test, y_train, y_test = train_test_split(fit,y,test_size = 0.3, stratify=y)


# In[14]:


#Normalize the data and PCA II
from sklearn import preprocessing
from sklearn.decomposition import PCA
pca=PCA(n_components=10)
XX=preprocessing.normalize(X)
pca.fit(XX)
fitx=pca.transform(XX)
## train-split after PCA II
Xx_train, Xx_test, yx_train, yx_test = train_test_split(fitx,y,test_size = 0.3, stratify=y)


# In[15]:


#from sklearn.neighbors import KNeighborsClassifier 
#knn=1
#knn = KNeighborsClassifier(n_neighbors=knn)
#knn.fit(X_train, y_train)
#scores= knn.score(X_test, y_test)
#print('scores for data X=:', scores)
#print('*'*50)
#knn.fit(Xx_train, yx_train)
#scores= knn.score(Xx_test, yx_test)
#print('scores for data XX=:', scores)
#print('*'*50)
#knn.fit(X0_train, y0_train)
#scores= knn.score(X0_test, y0_test)
#print('scores for data without pre-processing=:', scores)


# In[98]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(min_samples_split=2)
model.fit(X_train, y_train)
print('model score for X=', model.score(X_test, y_test))
print('*'*50)
model.fit(Xx_train, yx_train)
print('model score for XX=', model.score(Xx_test, yx_test))
print('*'*50)
model.fit(X0_train, y0_train)
print('scores for data without pre-processing=:', model.score(X0_test, y0_test))


# In[99]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(min_samples_split=3,n_estimators=500, max_features=.2)
model.fit(X_train, y_train)
print('model score for X=', model.score(X_test, y_test))
print('*'*50)
model.fit(Xx_train, yx_train)
print('model score for XX=', model.score(Xx_test, yx_test))
print('*'*50)
model.fit(X0_train, y0_train)
print('scores for data without pre-processing=:', model.score(X0_test, y0_test))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
modelada = RandomForestClassifier(max_depth=100, n_estimators=100, max_features=0.2)
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(base_estimator=modelada,n_estimators=300, random_state=0,learning_rate=1.5)
clf.fit(X_train,y_train)
print('model score for X=',clf.score(X_test,y_test))
print('*'*50)
clf.fit(Xx_train,yx_train)
print('model score for XX=',clf.score(Xx_test,yx_test))
print('*'*50)
clf.fit(X0_train,y0_train)
print('scores for data without pre-processing=:',clf.score(X0_test,y0_test))


# In[ ]:




