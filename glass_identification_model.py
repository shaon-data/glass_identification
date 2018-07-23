# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
import os

from scipy.stats import skew
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler,scale

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import accuracy_score

FILE_NAME = "data/glass.csv"
scores = []


def get_sys_exec_root_or_drive(current_path):
    path = current_path
    while os.path.split(path)[1]:
        path = os.path.split(path)[0]
    return path[:-1]



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





feature_names = ['id_number', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
class_names = ['building_windows_float_processed' , 'building_windows_non_float_processed' , 'vehicle_windows_float_processed' , 'containers' , 'tableware' , 'headlamps']


dot_data = "tree.dot"
export_graphviz(clf, out_file=dot_data,feature_names = feature_names, class_names = class_names)



current_path = os.path.dirname(os.path.abspath(__file__))
current_drive = get_sys_exec_root_or_drive(current_path)
os.system(current_drive)
os.system(current_path)
os.system("dot -Tpdf -O "+dot_data)



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
