
#Importing of libraries

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
#from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
#from operator import itemgetter
from imblearn.over_sampling import SMOTE


#importing data using pandas
df = pd.read_csv("training_data.csv")
df.replace(["low_bike_demand","high_bike_demand"],[0,1],inplace=True)
df_dup = df.drop(['snow','snowdepth','dew','day_of_week','holiday','precip'] , axis=1)
df_dup.head()


#train test splitting
X_train, X_test, y_train, y_test = train_test_split(df_dup.iloc[:,:9],df_dup.iloc[:,-1],random_state=1, test_size=0.3,shuffle=True)


#scaling the data points using Standard Scalar
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)



#implementing the Knn and fitting the train dataset
knn = KNeighborsClassifier(metric='euclidean')
knn.fit(X_train_scaled, y_train)


# getting the accuracy recall and precsion of the fitted model
y_pred = knn.predict(X_test_scaled)
precision = precision_score(y_test , y_pred)
recall = recall_score(y_test , y_pred)
f1 = f1_score(y_test,y_pred)

print('precision: ',precision)
print('recall: ',recall)
print('f1 score: ',f1)


knn.score(X_test_scaled , y_test)


# cross validating the model
score = cross_val_score(knn , X_train_scaled , y_train , cv=10, scoring='accuracy')
np.mean(score)


# Oversampling the dataset and doing train_test split
X = df_dup.drop(['increase_stock'] , axis = 1)
Y = df_dup['increase_stock']
X_resampled, y_resampled = SMOTE().fit_resample(X, Y)
X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(X_resampled , y_resampled,random_state=1, test_size=0.3,shuffle=True)


# standardizing the X columns
scalar = StandardScaler()
X_train_sample_scaled = scalar.fit_transform(X_train_sample)
X_test_sample_scaled = scalar.transform(X_test_sample)


# finding the best K value
k_values = [i for i in range (1,61)]
scores = []

for i in k_values:
    knn_loop = KNeighborsClassifier(n_neighbors=i)
    knn_loop.fit(X_train_sample_scaled , y_train_sample)
    scores.append((i,knn_loop.score(X_test_sample_scaled , y_test_sample)))

max = 0
for i in range(1,60):
    if max < scores[i][1]:
        max = scores[i][1]
        index = scores[i][0]
print(index , max )


# training the model with the optimum k with balanced data
knn_oversampled = KNeighborsClassifier(n_neighbors=2 , metric='euclidean')
knn_oversampled.fit(X_train_sample_scaled , y_train_sample)


# finding the recall, precision , accuracy
y_pred_2_sample = knn_oversampled.predict(X_test_sample_scaled)
precision = precision_score(y_test_sample , y_pred_2_sample)
recall = recall_score(y_test_sample , y_pred_2_sample)
f1 = f1_score(y_test_sample,y_pred_2_sample)

print('precision: ',precision)
print('recall: ',recall)
print('f1 score: ',f1)


knn_oversampled.score(X_test_sample_scaled , y_test_sample)
