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
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import time

wine = pd.read_csv("winequality-white.csv")
#remove density because the correlation between density and residual sugar is high and there is less correlation between density and quality, my classifier, then residual sugar and quality

plt.figure(figsize = (16, 7))

sns.heatmap(wine.corr(), annot = True, fmt = '0.2g', linewidths = 1)

plotnumber = 1

for col in wine:
    if plotnumber <= 12:
        ax = plt.subplot(4, 3, plotnumber)
        sns.distplot(wine[col])
        plt.xlabel(col, fontsize = 15)
        
    plotnumber += 1

grid_dict = {}
acc_dict = {}

wine = wine.drop('density', axis=1)
y = wine['quality']
X = wine.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y, random_state = 42)  


minmax = RobustScaler()
X_train = minmax.fit_transform(X_train)
X_test = minmax.transform(X_test)

for col in minmax:
    if plotnumber <= 12:
        ax = plt.subplot(4, 3, plotnumber)
        sns.distplot(minmax[col])
        plt.xlabel(col, fontsize = 15)
        
    plotnumber += 1

start_time = time.time()
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
param_grid = {'n_neighbors': [1, 2, 5, 10, 20],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 
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
plot_confusion_matrix(knn2, X_test, y_test, display_labels=y, xticks_rotation="vertical")


start_time = time.time()
svc = SVC()
svc.fit(X_train, y_train)
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'gamma':['scale', 'auto'],
              'kernel': ['linear', 'rbf']}  
   
grid = GridSearchCV(svc, param_grid, refit = True, verbose = 3,n_jobs=-1) 
grid.fit(X_train, y_train) 
grid_predictions = grid.predict(X_test) 
svc_acc1 = accuracy_score(y_test, svc.predict(X_test))
svc2 = SVC(C=10, gamma='scale', kernel='rbf')
svc2.fit(X_train, y_train)
svc_acc2 = accuracy_score(y_test, svc2.predict(X_test))
grid_dict["svc_acc"] = grid.best_params_
acc_dict["SVC"] = [svc_acc1, svc_acc2]
plot_confusion_matrix(svc2, X_test, y_test, display_labels=y, xticks_rotation="vertical")


start_time = time.time()
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
param_grid = {'criterion': ['gini', 'entropy'],   
              'splitter': ['best', 'random'], 
              'max_depth':[1, 2, 3, 4, 5, 7, 9]}  
   
grid = GridSearchCV(dt, param_grid, refit = True, verbose = 3,n_jobs=-1) 
grid.fit(X_train, y_train) 
grid_predictions = grid.predict(X_test)
dt_acc1 = accuracy_score(y_test, dt.predict(X_test))
dt2 = DecisionTreeClassifier(criterion='gini', max_depth=9, splitter='best')
dt2.fit(X_train, y_train)
dt_acc2 = accuracy_score(y_test, dt2.predict(X_test))
grid_dict["dtc_acc"] = grid.best_params_
acc_dict["DT"] = [dt_acc1, dt_acc2]
plot_confusion_matrix(dt2, X_test, y_test, display_labels=y, xticks_rotation="vertical")


start_time = time.time()
rf1 = RandomForestClassifier()
rf1.fit(X_train, y_train)
param_grid = {'n_estimators': [40, 50, 70, 80, 100],  
              'criterion': ['gini','entropy'], 
              'max_depth':[1, 2, 3, 4, 5, 7, 9]}  
   
grid = GridSearchCV(rf1, param_grid, refit = True, verbose = 3,n_jobs=-1) 
grid.fit(X_train, y_train) 
grid_predictions = grid.predict(X_test) 
rf_acc1 = accuracy_score(y_test, rf1.predict(X_test))
rf2 = RandomForestClassifier()
rf2.fit(X_train, y_train)
rf_acc2 = accuracy_score(y_test, rf2.predict(X_test))
grid_dict["rf_acc"] = grid.best_params_
acc_dict["RF"] = [rf_acc1, rf_acc2]
plot_confusion_matrix(rf2, X_test, y_test, display_labels=y, xticks_rotation="vertical")


start_time = time.time()
ada1 = AdaBoostClassifier(base_estimator = dt)
ada1.fit(X_train, y_train)
grid_param = {'n_estimators' : [40, 50, 70, 80, 100],
             'learning_rate' : [0.01, 0.1, 0.05, 0.5, 1, 10],
             'algorithm' : ['SAMME', 'SAMME.R']
}
grid = GridSearchCV(ada1, grid_param,  refit = True, verbose = 3,n_jobs=-1)
grid.fit(X_train, y_train)
ada_acc1 = accuracy_score(y_test, ada1.predict(X_test))
ada2 = AdaBoostClassifier(base_estimator = dt)
ada2.fit(X_train, y_train)
ada_acc2 = accuracy_score(y_test, ada2.predict(X_test))
grid_dict["ada_acc"] = grid.best_params_
acc_dict["ADA"] = [ada_acc1, ada_acc2]
plot_confusion_matrix(ada2, X_test, y_test, display_labels=y, xticks_rotation="vertical")



start_time = time.time()
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
param_grid = {'loss': ['deviance', 'exponential'],  
              'learning_rate': [0.01, 0.1, 0.05, 0.5, 1], 
              'n_estimators':[1, 10, 50, 100, 200]}  
   
grid = GridSearchCV(gb, param_grid, refit = True, verbose = 3,n_jobs=-1) 
grid.fit(X_train, y_train) 
grid_predictions = grid.predict(X_test) 
gb_acc = accuracy_score(y_test, gb.predict(X_test))
gb2 = GradientBoostingClassifier()
gb2.fit(X_train, y_train)
gb_acc2 = accuracy_score(y_test, gb2.predict(X_test))
grid_dict["gb_acc"] = grid.best_params_
acc_dict["GD"] = [gb_acc, gb_acc2]
plot_confusion_matrix(gb2, X_test, y_test, display_labels=y, xticks_rotation="vertical")


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
