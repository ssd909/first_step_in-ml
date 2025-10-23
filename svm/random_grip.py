import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import  loguniform

data={
    'weight' :[100,120,90,230,140,234,312,450,367,98],
    'brightness':[2,8,3,5,7,9,10,4,1,6],
     'round':[2,8,3,5,7,9,10,4,1,6],

    'target':['apple','orange','orange','apple','orange','apple','orange','orange','apple','apple'],

}
""
#label
data_from=pd.DataFrame(data=data)
data_from.to_csv('data_from.csv',index=False)
loaded_data=pd.read_csv('data_from.csv')
loaded_data['label']=loaded_data['target'].map({'apple':1,'orange':0})
loaded_data.drop('target',axis=1,inplace=True)
""
""
#scaling
feature_scaling=['brightness','weight','round']
ar=loaded_data[feature_scaling]
scaler=MinMaxScaler()
x_scaled=scaler.fit_transform(ar)
scaled_data=pd.DataFrame(data=x_scaled,columns=feature_scaling)
loaded_data[feature_scaling]=scaled_data
print(loaded_data)
""
""
#split data

y=loaded_data['label']
x=loaded_data.drop('label',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
""
svm_model=SVC()
test_grib={
    'C' :loguniform(1e-2,1e3),
    'gamma':loguniform(1e-4,1e-1),
    'kernel':['linear','rbf','sigmoid','poly'],

}
random_search=RandomizedSearchCV(
    estimator=svm_model,
    param_distributions=test_grib,
    scoring='accuracy',
    cv=4,
    verbose=2,
    n_jobs=-1,
    random_state=42
)
random_search.fit(x_train,y_train)
print(random_search.best_params_)
print(random_search.best_score_)