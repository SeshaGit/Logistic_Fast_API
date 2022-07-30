import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ALLOW_PRINT: bool = False

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid",color_codes = True)

data = pd.read_csv("banking.csv")

# exploring the data

if ALLOW_PRINT: print(data.head())
if ALLOW_PRINT: print(data.describe())

# Checking null values
if ALLOW_PRINT: print(data.isna().sum())

# Term deposit is determined via column 'y'

# Data Preprocessing & Manipulation
categorical_variables = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in categorical_variables:
    category_list = pd.get_dummies(data[var], prefix=var)
    data1 = data.join(category_list)
    data = data1
data_variables = data.columns.values.tolist()

# Removing Category variables as they are already replaced and identifying the columns to keep

columns_to_keep = [i for i in data_variables if i not in categorical_variables]

# Filtering the necessary columns
data = data[columns_to_keep]

# Doing a minority over sampling

from imblearn.over_sampling import SMOTE

X = data.loc[:,data.columns != 'y' ] # all independent variables with all rows

y = data.loc[:,data.columns == 'y'] #target variable with all rows
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 0)
over_sampling = SMOTE(random_state=0)
columns = X_train.columns
over_sample_X, over_sample_y = over_sampling.fit_resample(X_train, y_train)
over_sample_x = pd.DataFrame(data = over_sample_X, columns= columns)
over_sample_y = pd.DataFrame(data = over_sample_y, columns= ['y'])


data_final_vars=data.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg)
rfe = rfe.fit(over_sample_x, over_sample_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

# Selecting columns that add more value to the prediction outcome

cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no', 'default_unknown',
      'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar',
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"]
X=over_sample_x[cols]
y=over_sample_y['y']

import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(X[2])

# Log Regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)