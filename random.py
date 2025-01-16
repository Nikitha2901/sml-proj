#Importing the libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
#from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE


# Importing dataset using pandas
df = pd.read_csv("training_data.csv")
df.replace(["low_bike_demand","high_bike_demand"],[0,1],inplace=True)
df_dup = df.drop(['snow','snowdepth','dew','day_of_week','holiday','precip'] , axis=1)
df_dup.head()

# train test split
X_train, X_test, y_train, y_test = train_test_split(df_dup.iloc[:,:9],df_dup.iloc[:,-1],random_state=1, test_size=0.3,shuffle=True)

# Implementing random forest class and fitting the model
forest = RandomForestClassifier()
forest.fit(X_train , y_train)

# accuracy of the model
forest.score(X_test , y_test)


# finding recall prescision of the model
ypred = forest.predict(X_test)
precision = precision_score(y_true=y_test , y_pred=ypred)
recall = recall_score(y_true=y_test , y_pred=ypred)
f1 = f1_score(y_true=y_test , y_pred=ypred)

# cross validating the model
score = cross_val_score(forest , X_train , y_train , cv=10 , scoring='precision')
print(score)
print("%0.2f - max accuracy and %0.2f - average accuracy and %0.2f - min accuracy" %(score.max(), score.mean(), score.min()))


# Grid search parametes
params = { 'n_estimators':[100],
           'max_depth': [2, 3, 4, 5, 6,7,9,8,10,15,20],
           'random_state': [13],
           'criterion': ['gini','entropy','log_loss'],
           'min_samples_split': [2,3,4],
           'min_samples_leaf': [1,2,3,4],
           'max_features': ['sqrt','log2']}


rf = RandomForestClassifier()

# Implementing grid search
grid_search = GridSearchCV(rf , param_grid=params ,scoring='recall')
model = grid_search.fit(X_train,y_train)
print(model.best_params_)
model.score(X_test , y_test)



# Over sampling the dataset and applying the grid search
X = df_dup.drop(['increase_stock'] , axis = 1)
Y = df_dup['increase_stock']


X_resampled, y_resampled = SMOTE().fit_resample(X, Y)
X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(X_resampled , y_resampled,random_state=1, test_size=0.3,shuffle=True)


forest2 = RandomForestClassifier(criterion ='gini', max_depth=15, max_features='sqrt' , min_samples_leaf= 1, min_samples_split= 2, n_estimators= 100)
forest2.fit(X_train_sample , y_train_sample)
y_pred_sample = forest2.predict(X_test_sample)
precision_2 = precision_score(y_true=y_test_sample , y_pred=y_pred_sample)
recall_2 = recall_score(y_true=y_test_sample , y_pred=y_pred_sample)
score = forest2.score(X_test_sample , y_test_sample)
