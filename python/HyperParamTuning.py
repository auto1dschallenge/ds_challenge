from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import models


def RF_paramSearch(X,y):
    # Function to call GridSearchCV and find best parameters for a RandomForest     
    # model on the dataset X with labels y
    reg = models.get_rf_base()

    
    ''' 
    # Run grid search for max_depth and n_estimators
    param_grid = {"max_depth": [5,10,20,30],
                  "n_estimators": [10,20,30,50,70,100,150]}
    
    # run grid search
    grid_search = GridSearchCV(reg, param_grid=param_grid,cv=5)
    grid_search.fit(X, y.ravel())
    
    grid_search.best_params_
    '''
    # Returns n_estimators = 100, max_depth = 10
    
    param_dist = {"max_depth": [10],
                  "max_features": ["auto","sqrt","log2"],
                  "min_samples_split": [2, 5, 10],
                  "min_samples_leaf": [1, 2, 4],
                  "bootstrap": [True, False],
                  "criterion": ["mse"],
                  "n_estimators": [100]}
    
    # run randomized search
    
    grid_search2 = GridSearchCV(reg, param_grid=param_dist,cv=5)
    grid_search2.fit(X, y.ravel())

    return grid_search2.best_params_