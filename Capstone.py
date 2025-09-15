#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 17:49:20 2025

@author: andrewyan
"""

#import os
#os.chdir("Desktop/FML/Capstone")

#Seed my N-number
n_num = 16154377
import random
random.seed(n_num)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve

#Open the file
df = pd.read_csv('musicData.csv')

df_filtered = df.copy()
df_filtered = df_filtered.dropna()

#Convert any ? to NaNs
df_filtered.replace('?', np.nan, inplace=True)

#Encode the target variable Genre
le = LabelEncoder()
df_filtered['genre_encoded'] = le.fit_transform(df_filtered['music_genre'])
n_classes = len(le.classes_)
print(f"Number of Genres: {n_classes}")

#Stratified train-test split for each genre
test_index = []
for genre in df_filtered['music_genre'].unique():
    idx = df_filtered[df_filtered['music_genre'] == genre].sample(n=500, random_state=n_num).index
    test_index.extend(idx)

#Set the train and test datasets
test_df = df_filtered.loc[test_index]
train_df = df_filtered.drop(test_index)

#Set X and y train and test data, excluding information about artist/track
X_train = train_df.drop(columns=['instance_id', 'artist_name', 'track_name', 'obtained_date', 'music_genre', 'genre_encoded'])
y_train = train_df['genre_encoded']
X_test = test_df.drop(columns=['instance_id', 'artist_name', 'track_name', 'obtained_date', 'music_genre', 'genre_encoded'])
y_test = test_df['genre_encoded']

#Some variables are categorical, some are numerical
categorical_variables = ['key', 'mode']
numeric_variables = [col for col in X_train.columns if col not in categorical_variables]

#Preprocessing using Pipeline/ColumnTransformer for more simplified process
#SimpleImputer to compute missing values with strategies
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_variables),
    ('cat', categorical_pipeline, categorical_variables)
])

#Binarize y_test for multi-class AUC (shows correct/incorrect genres as 0-1)
y_test_bin = label_binarize(y_test, classes=range(n_classes))

#%%
#Run 3 models to compare them all: Random Forest, Decision Tree, and AdaBoost
#Model 1: Random Forest
#Build a Random Forest model using Pipeline for preprocessing, PCA, and the model
random_forest = Pipeline([
    ('rf_preprocessing', preprocessor),
    ('rf_pca', PCA(n_components=0.95, random_state=n_num)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=n_num))
])

#Fit on training set
random_forest.fit(X_train, y_train)

#Predict probabilities on test set
rf_y_pred_prob = random_forest.predict_proba(X_test)

#Compute AUC score
rf_auc = roc_auc_score(y_test_bin, rf_y_pred_prob)
print(f"Random Forest AUC Score: {rf_auc:.3f}")

#%%
#Model 2: Decision Tree
#Build a Decision Tree model using Pipeline for preprocessing, PCA, and the model
decision_tree = Pipeline([
    ('dt_preprocessing', preprocessor),
    ('dt_pca', PCA(n_components=0.95, random_state=n_num)),
    ('dt', DecisionTreeClassifier(random_state=n_num))
])

#Fit on training set
decision_tree.fit(X_train, y_train)

#Predict probabilities on test set
dt_y_pred_prob = decision_tree.predict_proba(X_test)

#Compute AUC score
dt_auc = roc_auc_score(y_test_bin, dt_y_pred_prob)
print(f"Decision Tree AUC Score: {dt_auc:.3f}")

#%%
#Model 3: AdaBoost
#Build an AdaBoost model using Pipeline for preprocessing, PCA, and the model
adaboost = Pipeline([
    ('adaboost_preprocessing', preprocessor),
    ('adaboost_pca', PCA(n_components=0.95, random_state=n_num)),
    ('adaboost_model', AdaBoostClassifier(random_state=n_num))
])

#Fit on training set
adaboost.fit(X_train, y_train)

#Predict probabilities on test set
adaboost_y_pred_prob = adaboost.predict_proba(X_test)

#Compute AUC score
adaboost_auc = roc_auc_score(y_test_bin, adaboost_y_pred_prob)
print(f"Decision Tree AUC Score: {adaboost_auc:.3f}")

#%%
#Plot all AUROC Curves
plt.figure(figsize=(8, 5))

#Flatten all values for micro-average
y_flatten = y_test_bin.ravel()
rf_scores = rf_y_pred_prob.ravel()
dt_scores = dt_y_pred_prob.ravel()
ab_scores = adaboost_y_pred_prob.ravel()

#Plot Random Forest Curve
rf_fpr, rf_tpr, rf_threshold = roc_curve(y_flatten, rf_scores)
plt.plot(rf_fpr, rf_tpr, color='blue', label=f'AUC: {rf_auc:.3f}')

#Plot Decision Tree Curve
dt_fpr, dt_tpr, dt_threshold = roc_curve(y_flatten, dt_scores)
plt.plot(dt_fpr, dt_tpr, color='green', label=f'AUC: {dt_auc:.3f}')

#Plot AdaBoost Curve
ab_fpr, ab_tpr, ab_threshold = roc_curve(y_flatten, ab_scores)
plt.plot(ab_fpr, ab_tpr, color='red', label=f'AUC: {adaboost_auc:.3f}')

#Plot Random Guess
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')

#Finish plot
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Micro-Average AUROC Curves")
plt.legend()
plt.show()
