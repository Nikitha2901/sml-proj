# Importing all the library
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from statistics import mean
from numpy import std #REVIEW
from numpy import arange #REVIEW

import sklearn.preprocessing as skl_pre
import sklearn.model_selection as skl_ms

from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold # For KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV #REVIEW

from sklearn.metrics import accuracy_score, precision_score, recall_score

import seaborn as sns

# Load the data and use panda to read the csv file provided
url = 'training_data.csv'
raw_data = pd.read_csv(url)
raw_data.replace (["low_bike_demand", "high_bike_demand"],[0,1], inplace=True)
raw_data_d = raw_data.drop(['snow', 'snowdepth', 'dew', 'day_of_week', 'holiday',
'precip'],axis=1)

#print(raw_data.iloc[:,-1])
# See the data details first with the followiing
#raw_data.info()
#raw_data.describe()

# Data preprocessing for categorical features
# This section is written to acknowledge that our project has both categorical and numerical
# In which, we have known that some features are binary (thus do not need to do encoding,
hot one or integer)

# Splitting the data into train and test data set where the train data is 70% of all dataset
np.random.seed(1)

trainI = np.random.choice(raw_data.shape[0], size=int(raw_data.shape[0]*0.7), replace=False)
trainIndex = raw_data.index.isin(trainI)
# print(trainIndex)

train = raw_data.iloc[trainIndex]
test = raw_data.iloc[~trainIndex]

# Define the train and test set, with the top 9 features from the data analysis
X_train = train[['hour_of_day', 'month', 'weekday', 'summertime', 'temp', 'humidity',
'windspeed', 'cloudcover', 'visibility']]
y_train = train['increase_stock']
X_test = test[['hour_of_day', 'month', 'weekday', 'summertime', 'temp', 'humidity',
'windspeed', 'cloudcover', 'visibility']]
y_test = test['increase_stock']

model = AdaBoostClassifier(learning_rate=0.1,
                   n_estimators=100)
print(model.get_params())
#print(model.get_params())
model.fit(X=X_train, y=y_train)
y_predict = model.predict(X_test)
print('Test error rate is %.3f' % np.mean(y_predict != y_test))
print(model.score(X_test , y_test))
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
print('Precision is %.3f'% prec)
print('Recall is %.3f'% rec)

VERBOSE=1

parameters = {
    'n_estimators': [10, 50, 100, 200],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1 , 1]
}
gscv = GridSearchCV(model, parameters, verbose=VERBOSE)
gscv.fit(X_train, y_train)
print(gscv.best_params_)
print(gscv.score(X_test , y_test))
