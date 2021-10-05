import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import time

wine = pd.read_csv("winequality-white.csv")
#remove density because the correlation between density and residual sugar is high and there is less correlation between density and quality, my classifier, then residual sugar and quality

plt.figure(figsize = (16, 7))

sns.heatmap(wine.corr(), annot = True, fmt = '0.2g', linewidths = 1)

grid_dict = {}
acc_dict = {}

wine = wine.drop('residual sugar', axis=1)
y = wine['quality']
X = wine.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y, random_state = 42)  


minmax = StandardScaler()
X_train = minmax.fit_transform(X_train)
X_test = minmax.transform(X_test)


start_time = time.time()
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
param_grid = {'n_neighbors': [1, 2, 5, 10, 20],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'weights' : ['uniform', 'distance']
                }

grid = GridSearchCV(knn, param_grid, refit = True, verbose = 3,n_jobs=-1) 
grid.fit(X_train, y_train) 
grid_predictions = grid.predict(X_test)
knn_acc1 = accuracy_score(y_test, knn.predict(X_test))
knn2 = KNeighborsClassifier(algorithm="auto", n_neighbors=1)
knn2.fit(X_train, y_train)
knn_acc2 = accuracy_score(y_test, knn2.predict(X_test))
grid_dict["knn_acc"] = grid.best_params_
acc_dict["KNN"] = [knn_acc1, knn_acc2]
plt.grid_search(grid.cv_results_, change='n_neighbours', kind='line')


start_time = time.time()
svc = SVC()
svc.fit(X_train, y_train)
param_grid = {'C': [80, 100, 130, 1000],  
              'gamma': [1, 0.1, 0.01, 10, 100], 
              'kernel': ['linear', 'rbf', 'sigmoid', 'poly']}  
   
grid = GridSearchCV(svc, param_grid, refit = True, verbose = 3,n_jobs=-1) 
grid.fit(X_train, y_train) 
grid_predictions = grid.predict(X_test) 
svc_acc1 = accuracy_score(y_test, svc.predict(X_test))
svc2 = SVC(C=100, gamma='scale', kernel='rbf')
svc2.fit(X_train, y_train)
svc_acc2 = accuracy_score(y_test, svc2.predict(X_test))
grid_dict["svc_acc"] = grid.best_params_
acc_dict["SVC"] = [svc_acc1, svc_acc2]
plt.grid_search(grid.cv_results_, change='kernel', kind='line')


start_time = time.time()
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
param_grid = {'criterion': ['gini', 'entropy'],   
              'splitter': ['best', 'random'], 
              'max_depth':[4, 5, 7, 9, 10, 11, None]}  
   
grid = GridSearchCV(dt, param_grid, refit = True, verbose = 3,n_jobs=-1) 
grid.fit(X_train, y_train) 
grid_predictions = grid.predict(X_test)
dt_acc1 = accuracy_score(y_test, dt.predict(X_test))
dt2 = DecisionTreeClassifier(criterion='gini', max_depth=9, splitter='random')
dt2.fit(X_train, y_train)
dt_acc2 = accuracy_score(y_test, dt2.predict(X_test))
grid_dict["dtc_acc"] = grid.best_params_
acc_dict["DT"] = [dt_acc1, dt_acc2]
plt.grid_search(grid.cv_results_, change='max_depth', kind='line')


start_time = time.time()
rf1 = RandomForestClassifier()
rf1.fit(X_train, y_train)
param_grid = {'n_estimators': [70, 80, 100, 130, 150],  
              'criterion': ['gini','entropy'], 
              'max_depth':[4, 5, 7, 9, 10, 11, None]}  
   
grid = GridSearchCV(rf1, param_grid, refit = True, verbose = 3,n_jobs=-1) 
grid.fit(X_train, y_train) 
grid_predictions = grid.predict(X_test) 
rf_acc1 = accuracy_score(y_test, rf1.predict(X_test))
rf2 = RandomForestClassifier(criterion='entropy', max_depth=9, n_estimators=80)
rf2.fit(X_train, y_train)
rf_acc2 = accuracy_score(y_test, rf2.predict(X_test))
grid_dict["rf_acc"] = grid.best_params_
acc_dict["RF"] = [rf_acc1, rf_acc2]
plt.grid_search(grid.cv_results_, change='max_depth', kind='line')



start_time = time.time()
ada1 = AdaBoostClassifier(base_estimator = dt)
ada1.fit(X_train, y_train)
grid_param = {'n_estimators' : [40, 50, 60, 65, 70, 80, 100],
             'learning_rate' : [0.01, 0.1, 0.05, 0.5, 1, 10],
             'algorithm' : ['SAMME', 'SAMME.R']
}
grid = GridSearchCV(ada1, grid_param,  refit = True, verbose = 3,n_jobs=-1)
grid.fit(X_train, y_train)
ada_acc1 = accuracy_score(y_test, ada1.predict(X_test))
ada2 = AdaBoostClassifier(base_estimator = dt, algorithm='SAMMER.R', learning_rate=0.05, n_estimators=70)
ada2.fit(X_train, y_train)
ada_acc2 = accuracy_score(y_test, ada2.predict(X_test))
grid_dict["ada_acc"] = grid.best_params_
acc_dict["ADA"] = [ada_acc1, ada_acc2]
plt.grid_search(grid.cv_results_, change='n_estimators', kind='line')



start_time = time.time()
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
param_grid = {'loss': ['deviance', 'exponential'],  
              'learning_rate': [0.01, 0.2, 0.3, 0.1, 0.05, 0.5, 1], 
              'n_estimators':[100, 200, 220, 230, 240, 250],
              'max_features' : ['auto', 'sqrt', 'log2']}  
   
grid = GridSearchCV(gb, param_grid, refit = True, verbose = 3,n_jobs=-1) 
grid.fit(X_train, y_train) 
grid_predictions = grid.predict(X_test) 
gb_acc = accuracy_score(y_test, gb.predict(X_test))
gb2 = GradientBoostingClassifier(learning_rate=0.01, loss='deviance', n_estimators=200)
gb2.fit(X_train, y_train)
gb_acc2 = accuracy_score(y_test, gb2.predict(X_test))
grid_dict["gb_acc"] = grid.best_params_
acc_dict["GD"] = [gb_acc, gb_acc2]
plt.grid_search(grid.cv_results_, change='n_estimators', kind='line')



print("-----------BEST HYPER-PARAMETERS-----------")
print(grid_dict)

print("-----------BEFORE AND AFTER ACCUARACY-----------")
print(acc_dict)

models_pre = pd.DataFrame({
    'Model' : ['KNN', 'SVC', 'Decision Tree', 'Random Forest','Ada Boost','Gradient Boosting'],
    'Score' : [knn_acc1, svc_acc1, dt_acc1, rf_acc1, ada_acc1, gb_acc]
})

models_post =  pd.DataFrame({
    'Model' : ['KNN', 'SVC', 'Decision Tree', 'Random Forest','Ada Boost','Gradient Boosting'],
    'Score' : [knn_acc2, svc_acc2, dt_acc2, rf_acc2, ada_acc2, gb_acc2]
})

plt.figure(figsize = (20, 8))
sns.lineplot(x = 'Model', y = 'Score', data = models_pre)

plt.figure(figsize = (20, 8))
sns.lineplot(x = 'Model', y = 'Score', data = models_post)
plt.show()
