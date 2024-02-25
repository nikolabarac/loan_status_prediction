# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 14:03:24 2024

@author: Nikola
"""
#Importing libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#Importing data

df = pd.read_csv('D:/Python_e/loan_status_prediction/loan_data.csv')


#Removing rows with missing data

df1 = df.dropna(subset=['Credit_History'], inplace = False)

df1.dropna(subset=['Gender'], inplace = True)

df1.dropna(subset=['Self_Employed'], inplace = True)

df1.dropna(subset=['Loan_Amount_Term'], inplace = True)

df1.dropna(subset=['Dependents'], inplace = True)

df1.info()


#Making stratified train and test set

from sklearn.model_selection import train_test_split

strat_train_set, strat_test_set = train_test_split(
df1, test_size=0.2, stratify=df1["Credit_History"], random_state=42)

strat_train_set['Credit_History'].value_counts()/len(strat_train_set)

strat_test_set['Credit_History'].value_counts()/len(strat_test_set)

