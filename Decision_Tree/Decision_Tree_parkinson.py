#!/usr/bin/env python
# coding: utf-8

# In[33]:


# Identificar doença de Parkinson a partir de análise vocal
# Dataset: https://archive.ics.uci.edu/ml/datasets/Parkinsons

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[78]:


dataset = pd.read_csv('parkinsons.data')

data = dataset.iloc[:, 1:18]

X = data.iloc[:, :16]
y = data.iloc[:,16].values


# In[79]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[80]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


# In[81]:


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


# In[ ]:




