#importing all library function
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,recall_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

#to remove warnings in output
import warnings
warnings.filterwarnings("ignore")

#use panda to read the csv file provided,csv file is stored in my local machine.
data = pd.read_csv('../sml-proj/training_data.csv')

#to check how data in printed, first 5 rows with columns will printed.
data.head() 

#replacing low_bike_demand and high_bike_demand with 0 and 1
data.replace(["low_bike_demand","high_bike_demand"],[0,1],inplace=True)

#dropping the featutes that is not needed
data_new=data.drop(["precip","holiday","snow","snowdepth","day_of_week","dew"],axis=1)

#storing the data in variable X and Y 
X=data_new.iloc[:,:9] 
Y=data_new.iloc[:,-1] 

#splitting the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=1,test_size=0.2, shuffle=True) 

#using logisticregression model function, fitting the X_train and Y_train data into it
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

#using score method to calculate the accuracy of X_test and Y_test after training the model
logreg.score(X_test,Y_test)

#performing crossvalidation of data then finding the accurancy,precision and recall value and F1_score with unsampled data features
C = [0.001,0.01,0.1,1,10,100,1000]
penalty = [ 'l2' , 'l1']
solver = [ 'newton-cg','lbfgs','sag']
max_iter =[100,1000,10000]
randomgrid ={'C': C ,'penalty': penalty , 'solver' : solver ,  'max_iter': max_iter }
logreg1 = LogisticRegressionCV(penalty='l2',solver='lbfgs',max_iter=100,Cs=10)
logreg1.fit(X_train, Y_train)
y_pred = logreg.predict(X_test)
y_pred = logreg1.predict(X_test)

Precision = metrics.precision_score(Y_test,y_pred)
Precision

Sensitivity_recall = metrics.recall_score(Y_test,y_pred)
Sensitivity_recall

F1_score = metrics.f1_score(Y_test,y_pred)
F1_score

#performing gridsearch hyper_parametre tuning of data then finding the accurancy,precision and recall value and F1_score with unsampled dataset
C = [0.001,0.01,0.1,1,10,100,1000]
penalty = [ 'l2']
solver = [ 'newton-cg','lbfgs','sag']
max_iter =[100,1000,10000]
randomgrid ={'C': C ,'penalty': penalty , 'solver' : solver ,  'max_iter': max_iter }
grid_search = GridSearchCV(logreg,randomgrid, cv=10, scoring='accuracy')

grid_search.fit(X_train, Y_train)
grid_search.score(X_test, Y_test)

y_pred1=grid_search.predict(X_test)
metrics.recall_score(Y_test,y_pred1)
metrics.precision_score(Y_test,y_pred1)

#oversampling the data using smote
X_resampled, y_resampled = SMOTE().fit_resample(X, Y)

y_pred_2 = logreg_hyper_2.predict(X_test_scaled_2)

#splitting the data after sampling
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_resampled,y_resampled,random_state=1, test_size=0.3,shuffle=True)

#normalizing the dataset
scalar = StandardScaler()
X_train_scaled_2 = scalar.fit_transform(X_train_2)
X_test_scaled_2 = scalar.fit_transform(X_test_2)

#Performing crossvalidation with sampled dataset and calculating accuracy,precison,recall
logreg_2 = LogisticRegressionCV(Cs=10 , max_iter= 100 , penalty='l2' , solver='newton-cg')
logreg_2.fit(X_train_scaled_2, y_train_2)

accuracy = logreg2.score(X_test_scaled_2 , y_test_2)
accuracy

y_pred_2 = logreg_hyper_2.predict(X_test_scaled_2)
metrics.recall_score(y_test_2,y_pred_2)
metrics.precision_score(y_test_2,y_pred_2)


performing gridsearch hyper_parametre tuning of data then finding the accurancy,precision and recall value and F1_score with the sampled dataset
C = [0.001,0.01,0.1,1,10,100,1000]
penalty = [ 'l2' , 'l1']
solver = [ 'newton-cg','lbfgs','sag']
max_iter =[100,1000,10000]
randomgrid ={'C': C ,'penalty': penalty , 'solver' : solver ,  'max_iter': max_iter }
logreg_hyper=LogisticRegression()
grid_search = GridSearchCV(logreg_hyper,randomgrid, cv=10, scoring='accuracy')
grid_search.fit(X_train_scaled_2, y_train_2)

accuracy = logreg_hyper_2.score(X_test_scaled_2 , y_test_2)
accuracy

y_pred_2 = logreg_hyper_2.predict(X_test_scaled_2)
metrics.recall_score(y_test_2,y_pred_2)
metrics.precision_score(y_test_2,y_pred_2)