# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
from scipy.stats import skew
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler,scale

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

FILE_NAME = "data/glass.csv"
scores = []

for c in range(100):
    data = pd.read_csv(FILE_NAME,names=['id_number','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','class'])
    data = shuffle(data)

        
    features = data.ix[:,:-1]
    target = data.ix[:,-1]

    X_train , X_test, y_train, y_test = train_test_split( features, target, test_size = 0.3, random_state = 100)
    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    scores.append( accuracy_score(y_test,y_pred)*100 )
    

scores = np.array(scores)
average = np.mean(scores)
median = np.median(scores)
mode = st.mode(scores)
skew = skew(scores)

print("Session:",len(scores))
print("Average:",average)
print("Median:",median)
print("Mode:  ",mode)
print("Skew to l:",skew)
print("Min:",min(scores))
print("Max:",max(scores))


plt.hist(sorted(scores),edgecolor='k',alpha=0.65)
plt.axvline(st.mode(scores), color='k', linestyle='dashed', linewidth=1, label="Mode")
plt.axvline(np.median(np.array(scores)), color='g', linestyle='dashed', linewidth=1, label = "Median")
plt.axvline(sum(scores)/len(scores), color='r', linestyle='dashed', linewidth=1, label = "Average")
plt.legend()
plt.show()
