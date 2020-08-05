# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:03:52 2020

@author: avatsyay
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
"""%matplotlib inline"""

import warnings
warnings.filterwarnings('ignore')

Heart=pd.read_csv(r'C:\Users\avatsyay\Documents\Aman\Edu Material\ML-AI\Hackathon\ML-MT-WebApp-master\heart.csv')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
X=Heart.drop(['target'],axis=1)
y=Heart['target']
X_train, X_test, y_train, y_test =train_test_split(X,y,
                                                   test_size=0.25,
                                                   random_state=0,
                                                   )

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

Hyperparameter={"C":[0.05,1,1.5,2,2.5,3], "penalty":["l1","l2"]}
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,Hyperparameter,cv=10)
logreg_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

import pickle
# save the model to disk
filename = 'Heart_model.pkl'
pickle.dump(logreg_cv, open(filename, 'wb'))

