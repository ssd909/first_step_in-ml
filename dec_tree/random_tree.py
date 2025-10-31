import numpy as np
from scipy.stats import bootstrap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris=load_iris()

x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

tree_model=RandomForestClassifier(

    random_state=42,



)
param_grid={
    'n_estimators':[10,20,30,40,50,100],
    'criterion':['gini','entropy'],
    'max_depth':[2,3,4,5,6],
    'bootstrap':[True,False],
    'min_samples_split':[2,3,4,5,6],

}
grid_search=GridSearchCV(
   estimator= tree_model,
   param_grid= param_grid,
    scoring='recall_weighted',
    cv=10,
    n_jobs=-1,

)
grid_search.fit(x_train,y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
train_model=RandomForestClassifier(
    random_state=42,
    n_estimators=grid_search.best_params_['n_estimators'],
    criterion=grid_search.best_params_['criterion'],
    max_depth=grid_search.best_params_['max_depth'],
    min_samples_split=grid_search.best_params_['min_samples_split'],
   bootstrap=grid_search.best_params_['bootstrap'],
)
train_model.fit(x_train,y_train)
y_pred=train_model.predict(x_test)
classification_report=classification_report(y_test,y_pred)
print(classification_report)




