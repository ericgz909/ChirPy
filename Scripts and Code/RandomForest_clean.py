#!/usr/bin/env python
# coding: utf-8

# ### Import Packages

import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

##################################  concatenating CSV files ###############

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

##################################### Filtering dataframe to reduce its size  #############

df=pd.read_csv('dataframe.csv')


Z=df.primary_label.value_counts()[0:10]
l=Z.index

for i in range(0,10):    
    if i==0:
        filter_df=df[df['primary_label']==l[i]]
    else:
        filter_df=filter_df.append(df[df['primary_label']==l[i]],ignore_index=True)
DF=filter_df.dropna()
 
########################### Feature columns and target column ####################



X=DF[['peak0_accuracy', 'peak0_freqency', 'peak0_prominence','peak0_width', 'peak0_width_height',
      'peak1_accuracy', 'peak1_freqency', 'peak1_prominence', 'peak1_width', 'peak1_width_height',
       'peak2_accuracy', 'peak2_freqency', 'peak2_prominence', 'peak2_width','peak2_width_height']]
Y=DF['primary_label']


y = Y.astype("category").cat.codes

################################ Train_split for original data #############


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, stratify=y)


################################# Normalizing the data and PCA ###############

from sklearn import preprocessing
from sklearn.decomposition import PCA
pca=PCA(n_components=10)
XX=preprocessing.normalize(X)
pca.fit(XX)
fit=pca.transform(XX)
#Train_split for normalized data
Xx_train, Xx_test, yx_train, yx_test = train_test_split(fit,y,test_size = 0.3, stratify=y)

################################ DecisionTreeClassifier ##################


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(min_samples_split=2)
model.fit(X_train, y_train)
print('*'*20,'DecisionTreeClassifier','*'*20 )
print()
print('Model score for  original data X =', model.score(X_test, y_test))
model.fit(Xx_train, yx_train)
print('Model score for normalized data XX =', model.score(Xx_test, yx_test))
print('\n')



############################## RandomForestClassifier ###################


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(min_samples_split=3,n_estimators=500, max_features=.2)
model.fit(X_train, y_train)
print('*'*20,'RandomForestClassifier','*'*20 )
print()
print('Model score for  original data X =', model.score(X_test, y_test))
model.fit(Xx_train, yx_train)
print('Model score for normalized data XX =', model.score(Xx_test, yx_test))
print('\n')
##########################   AdaBoostClassifier  ######################

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
modelada = RandomForestClassifier(max_depth=100, n_estimators=100, max_features=0.2)

clf = AdaBoostClassifier(base_estimator=modelada,n_estimators=300, random_state=0,learning_rate=1.5)
clf.fit(X_train,y_train)
print('*'*20,'AdaBoostClassifier','*'*20 )
print()
print('Model score for  original data X =',clf.score(X_test,y_test))
clf.fit(Xx_train,yx_train)
print('Model score for normalized data XX =',clf.score(Xx_test,yx_test))
print('\n')





