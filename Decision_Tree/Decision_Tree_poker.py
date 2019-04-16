#!/usr/bin/env python
# coding: utf-8
# Prever as mãos do poker
# Dataset: https://archive.ics.uci.edu/ml/datasets/Poker+Hand

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv('poker_hand.csv')

X = dataset.iloc[:, :10].values
y = dataset.iloc[:,10].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

t = cm.trace() #Soma da diagonal principal (resultados verdadeiros)
total = cm.sum() #Soma de todos os elementos da matriz

# Cálculo da acurária pela confusion matrix
acc = (float(t)/total)*100
print "Accuracy CM: %.2f%%"%acc


# Cálculo da acurária pelo método score
score = clf.score(X_test, y_test)*100
print "Score: %.2f%%"%score
