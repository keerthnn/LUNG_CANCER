# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 22:08:12 2020

@author: DELL
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import pickle

data = pd.read_csv(r'lung_data.csv')
x = data.iloc[:, 0:15]
y = data.variables
y = y.map({'YES':1,'NO':0})


data.GENDER.replace(['M','F'], [1, 0], inplace=True)
data.LUNG_CANCER.replace(['YES','NO'], [1, 0], inplace=True)

y = np.array(data.variables.tolist())
data = data.drop('variables', 1)
X = np.array(data.to_numpy())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
params = {'criterion':['entropy'],
          'n_estimators':[10],
          'min_samples_leaf':[1],
          'min_samples_split':[3], 
          'random_state':[123],
          'n_jobs':[-1]}
model1 = GridSearchCV(classifier, param_grid=params, n_jobs=-1)
model1.fit(X_train,y_train)
pickle.dump(model1,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
