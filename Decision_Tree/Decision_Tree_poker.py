#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Prever as mãos do poker
# Dataset: https://archive.ics.uci.edu/ml/datasets/Poker+Hand

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[129]:


'''
Merge das bases de dados de treinamento e test

dataset_training = pd.read_csv('poker-hand-training-true.data')
dataset_test = pd.read_csv('poker-hand-testing.data')

frames = [dataset_training, dataset_training]
dataset = pd.concat(frames)
dataset.to_csv('poker_hand.csv', index=False)
'''


# In[130]:


dataset = pd.read_csv('poker_hand.csv')

X = dataset.iloc[:, :10].values
y = dataset.iloc[:,10].values


# In[131]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[132]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


# In[133]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

t = cm.trace() #Soma da diagonal principal (resultados verdadeiros)
total = cm.sum() #Soma de todos os elementos da matriz

# Cálculo da acurária pela confusion matrix
acc = (float(t)/total)*100
print "Accuracy CM: %.2f%%"%acc


# Cálculo da acurária pelo método score
score = clf.score(X_test, y_test)*100
print "Score: %.2f%%"%score


# In[2]:


'''
import graphviz
from sklearn import tree

feature_names = dataset.iloc[:, :10].columns.values
target_names = ['0', '1', '2', '3','4','5','6', '7', '8', '9']

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("poker_hand") 
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_names, 
                                class_names=target_names, filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph
'''


# In[ ]:




