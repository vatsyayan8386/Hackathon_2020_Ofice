# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:18:35 2020

@author: avatsyay
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

Diabetes=pd.read_csv(r'C:\Users\avatsyay\Documents\Aman\Edu Material\ML-AI\Hackathon\ML-MT-WebApp-master\diabetes.csv')

Diabetes['Pregnancies']=np.where(Diabetes['Pregnancies']==0,Diabetes['Pregnancies'].median(),Diabetes['Pregnancies'])
Diabetes['BloodPressure']=np.where(Diabetes['BloodPressure']==0,Diabetes['BloodPressure'].median(),Diabetes['BloodPressure'])
Diabetes['SkinThickness']=np.where(Diabetes['SkinThickness']==0,Diabetes['SkinThickness'].median(),Diabetes['SkinThickness'])
Diabetes['Insulin']=np.where(Diabetes['Insulin']==0,Diabetes['Insulin'].median(),Diabetes['Insulin'])
Diabetes['BMI']=np.where(Diabetes['BMI']==0,Diabetes['BMI'].median(),Diabetes['BMI'])
Diabetes['Glucose']=np.where(Diabetes['Glucose']==0,Diabetes['Glucose'].median(),Diabetes['Glucose'])
Diabetes.head() 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(Diabetes.drop('Outcome',axis=1), Diabetes['Outcome'], test_size=0.15, 
                                                    random_state=500)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

from scipy.stats import uniform
model=LogisticRegression()
Penalty = ['l1','l2']
c = uniform()
hyperparameters=dict(C=c,penalty=Penalty)

from sklearn.model_selection import RandomizedSearchCV


logreg_RndmCV=RandomizedSearchCV(model,Hyperparameter,cv=10)    
logreg_RndmCV.fit(X_train,y_train)
"""Prediction_LR = LR_randomsearch.BestModelPredict(X_train)"""


"""Prediction_log_new = LR_randomsearch.predict(X_test)"""

print("tuned hpyerparameters :(best parameters) ",logreg_RndmCV.best_params_)
print("accuracy :",logreg_RndmCV.best_score_)


import pickle
# save the model to disk
filename = 'Diabetes_model.pkl'
pickle.dump(logreg_RndmCV, open(filename, 'wb'))

