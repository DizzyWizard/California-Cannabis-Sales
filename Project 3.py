#!/usr/bin/env python
# coding: utf-8

# ## 1.Merge 2.Time Series Engineering and 3.adding new features

# In[2]:


import sklearn.metrics.cluster as smc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np 
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import decomposition, model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingRegressor
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from itertools import combinations
import statsmodels.api as sm
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.svm import SVC


# In[3]:


brandTotalSales_df = pd.read_csv(os.path.join('data','BrandTotalSales.csv'))
brandTotalUnits_df = pd.read_csv(os.path.join('data','BrandTotalUnits.csv'))
brandAverageRetailPrice_df = pd.read_csv(os.path.join('data','BrandAverageRetailPrice.csv'))
brandDetails_df = pd.read_csv(os.path.join('data','BrandDetails.csv'))


# Check the meaning of columns.

# In[4]:


brandDetails_df.info()
brandTotalUnits_df.info()
brandTotalSales_df.info()
brandAverageRetailPrice_df.info()


# Convert months to datetime.

# In[5]:


brandTotalSales_df['Months'] = pd.to_datetime(brandTotalSales_df['Months'])
brandTotalUnits_df['Months'] = pd.to_datetime(brandTotalUnits_df['Months'])
brandAverageRetailPrice_df['Months'] = pd.to_datetime(brandAverageRetailPrice_df['Months'])


# Convert too large total units to float, trim it first then convert.

# In[6]:


brandTotalSales_df['Total Sales ($)'] = brandTotalSales_df['Total Sales ($)'].str[:8]
brandTotalSales_df['Total Sales ($)'] = brandTotalSales_df['Total Sales ($)'].str.replace(',', '')
brandTotalSales_df['Total Sales ($)'] = pd.to_numeric(brandTotalSales_df['Total Sales ($)'])


brandTotalSales_df.info()


# In[7]:


brandTotalUnits_df['Total Units'] = brandTotalUnits_df['Total Units'].str[:8]
brandTotalUnits_df['Total Units'] = brandTotalUnits_df['Total Units'].str.replace(',', '').astype(float)
brandTotalUnits_df['Total Units'] = pd.to_numeric(brandTotalUnits_df['Total Units'])


brandTotalUnits_df.info()


# Break the datasets up bt brands

# In[8]:


brands = brandTotalUnits_df['Brands'].unique()
brands


# In[9]:


l = len(brands)
tempDf = []
featureEngineeredDf = pd.DataFrame()
for i, brand in enumerate(brands):
    newDf = brandTotalUnits_df.where(brandTotalUnits_df.Brands==brand)
    newDf.loc[:,'Previous Month Units'] = newDf.loc[:,'Total Units'].shift(1)
    
    newDf = newDf.merge( brandTotalSales_df[brandTotalSales_df.Brand == brand], left_on='Months',right_on='Months')
    newDf = newDf.merge( brandAverageRetailPrice_df[brandAverageRetailPrice_df.Brands == brand], left_on='Months',right_on='Months')
    
    newDf.loc[:,'Current Unit Averages'] = (newDf.loc[:,'Total Units'].shift(1) +newDf.loc[:,'Total Units'].shift(2) + newDf.loc[:,'Total Units'].shift(3) + newDf.loc[:,'Total Units'].shift(4)) / 4
    
    newDf = newDf.drop(['Brands_x'], 1)
    newDf = newDf.drop(['Brands_y'], 1)

    inhaleables = 0
    topicals = 0
    ingestibles = 0
    if 'Inhaleables' in brandDetails_df[brandDetails_df.Brand == brand]['Category L1'].values:
        inhaleables = 1
    if 'Topicals' in brandDetails_df[brandDetails_df.Brand == brand]['Category L1'].values:
        topicals = 1
    if 'Ingestibles' in brandDetails_df[brandDetails_df.Brand == brand]['Category L1'].values:
        ingestibles = 1
    newDf['Inhaleables'] = inhaleables
    newDf['Topicals'] = topicals
    newDf['Ingestibles'] = ingestibles
    
    
    newDf['ProdCount'] = len(brandDetails_df.loc[brandDetails_df['Brand'] == brand])
    
    tempDf.append(newDf)
    
featureEngineeredDf = pd.concat(tempDf)


# In[10]:


featureEngineeredDf.info()


# Drop the brands that lacks too much data(more than 6 months), has no information about them(all nans) or have no products(ProdCount == 0).

# In[11]:


l = len(brands)
for i,brand in enumerate(brands):
    tempDf = featureEngineeredDf[featureEngineeredDf.Brand == brand]
    lTempDf = len(tempDf)
    
    #Lacks too much data
    if lTempDf <= 6:
        featureEngineeredDf = featureEngineeredDf[featureEngineeredDf.Brand != brand]
        continue
    
    #all nans
    for column in featureEngineeredDf.columns[featureEngineeredDf.isna().any()].tolist():
        if len(list(tempDf[column].unique())) == 1:
            featureEngineeredDf = featureEngineeredDf.loc[featureEngineeredDf.Brand != brand]

#Total Number of products is 0
featureEngineeredDf = featureEngineeredDf.loc[featureEngineeredDf['ProdCount'] != 0]


# In[12]:


#Replace null values with the median of all values
brands = list(featureEngineeredDf['Brand'].unique())
l = len(brands)
for i, brand in enumerate(brands):
    for column in featureEngineeredDf.columns[featureEngineeredDf.isna().any()].tolist():
        median = featureEngineeredDf.loc[featureEngineeredDf.Brand==brand,column].median()
        featureEngineeredDf.loc[featureEngineeredDf['Brand'] == brand, column] = featureEngineeredDf.loc[featureEngineeredDf['Brand'] == brand,column].fillna(value = median)
featureEngineeredDf.info()
featureEngineeredDf.head(10)


# ### 4. Linear Regression

# A useful helper function

# In[13]:


def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))    
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))


# In[14]:


numericalFeatures = ['Previous Month Units',
                     'Inhaleables',
                     'Topicals',
                     'Ingestibles',
                     'ProdCount',
                    'Current Unit Averages']
categoricalFeatures = ['Brand']

# drop features
featuresToDrop = list(set(featureEngineeredDf.columns) - set(categoricalFeatures) - set(numericalFeatures))
tempDf = featureEngineeredDf.drop(featuresToDrop, axis=1).copy(deep=True)
tempDf.info()


# In[15]:


# pipeline
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, numericalFeatures),
    ('cat', OneHotEncoder(), categoricalFeatures),
])

preparedData = full_pipeline.fit_transform(tempDf).toarray()

# train test split
y = featureEngineeredDf['Total Sales ($)'].copy()
train_X, test_X, train_Y, test_Y = train_test_split(preparedData,y, test_size=0.3, random_state=42)

# linear regresssion
linearRegression = LinearRegression()
linearRegression.fit(train_X, train_Y)

trainPrediction = linearRegression.predict(train_X)
testPrediction = linearRegression.predict(test_X)


# In[16]:


regression_results(test_Y, testPrediction)


# In[17]:


sm_x = sm.add_constant(preparedData)


# ### 5. PCA

# In[18]:


pca = PCA(n_components = 50)
principleComponents = pca.fit_transform(preparedData)

# train test split
labels = featureEngineeredDf['Total Sales ($)'].copy()
train_X, test_X, train_Y, test_Y = train_test_split(principleComponents, y, test_size=0.3, random_state=42)

#linear regression
linearRegression.fit(train_X, train_Y)

trainPrediction = linearRegression.predict(train_X)
testPrediction = linearRegression.predict(test_X)


# In[19]:


regression_results(test_Y,testPrediction)


# ### 6.Ensemble Methods (I used Random Forest)

# In[20]:


labels = featureEngineeredDf['Total Sales ($)'].copy()
train_X,test_X,train_Y,test_Y = train_test_split(preparedData,y,test_size=0.3,random_state=42)

randomForest = RandomForestRegressor(max_depth=12, random_state=42)
randomForest.fit(train_X,train_Y)

trainPrediction = randomForest.predict(train_X)
testPrediction = randomForest.predict(test_X)


# In[21]:


regression_results(test_Y,testPrediction)


# ### 7.Cross Validation

# In[22]:


kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle=True) 
print("building models...")
linear_model_kfold = LinearRegression()
rf_model_kfold = RandomForestRegressor(max_depth=12, random_state=42)
print("generating results...")
linear_results_kfold = model_selection.cross_val_score(linear_model_kfold, train_X, train_Y, cv=kfold)
rf_results_kfold = model_selection.cross_val_score(rf_model_kfold, train_X, train_Y, cv=kfold)
print("done!")

# Because we're collecting results from all runs, we take the mean value
print("Linear Regression Accuracy: %.2f%%" % (linear_results_kfold.mean()*100.0)) 

print("Random Forest Accuracy: %.2f%%" % (rf_results_kfold.mean()*100.0)) 


# In[23]:


rf_model_kfold.fit(train_X, train_Y)
rfPred= rf_model_kfold.predict(test_X)
regression_results(test_Y,rfPred)


# In[24]:


linear_model_kfold.fit(train_X, train_Y)
lgPred = linear_model_kfold.predict(test_X)
regression_results(test_Y,lgPred)


# ### 8. Grid Search
# 
# Since Random Forest has better performance, I chose it to be my predictive model.

# In[25]:


max_depth=[8]
n_estimators = [50,100,200]
n_jobs = [-1]
param_grid = dict(
    max_depth=max_depth,
    n_estimators=n_estimators,
    n_jobs=n_jobs
)

randomForest = RandomForestRegressor()
gridSearchCV = GridSearchCV(
    estimator=randomForest,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=10
)

gridResult = gridSearchCV.fit(preparedData,y)
print('Negative mae:', gridResult.best_score_)
print('Hyperparameters:', gridResult.best_params_)


# ### 9. Best Model

# In[29]:


labels = featureEngineeredDf['Total Sales ($)'].copy()
train_X,test_X,train_Y,test_Y = train_test_split(preparedData,y,test_size=0.3,random_state=42)

kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle=True) 
rf_model_kfold = RandomForestRegressor(max_depth=10, random_state=42)
rf_model_kfold.fit(train_X, train_Y)
rfPred= rf_model_kfold.predict(test_X)
regression_results(test_Y,rfPred)

