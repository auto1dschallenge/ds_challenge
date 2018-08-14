from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV,Lasso
from sklearn.pipeline import make_pipeline
import pandas as pd
from itertools import compress
import seaborn as sns

def lasso_coefficients(X,y):
    ## Use lasso regression to select "best" features from dataset X
    cols = X.columns
    # scale features for regression
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)
    sc_y = StandardScaler()
    tmp = sc_y.fit_transform(y)
    y= tmp

    lasso = LassoCV(random_state = 0)
    sfm = SelectFromModel(lasso)
    sfm.fit(X,y)
    
    ic = sfm.get_support()
    new_cols = list(compress(cols,ic))
    
    return new_cols


def correlation_cols(X):
    # Return columns derived from manual feature selection as described in the documentation
    drop_cont_cols = ['symboling', 'normalized-losses', 'height', 'stroke', 'compression-ratio', 'peak-rpm', 'city-mpg', 'highway-mpg']
    drop_cat_cols = ['num-of-doors']
    for i in X.columns:
        if i in drop_cont_cols+drop_cat_cols:
            X=X.drop(i,axis=1)
    return X.columns
    
    
    return 

def rf_feature_importance(model,x):
    # return Feature Importance for a RandomForest Model model and dataset x
    importance=sorted(zip(x.columns,model.feature_importances_),key=lambda x: -x[1])
    labels = ['feature','importance']
    fi = pd.DataFrame.from_records(importance,columns=labels)
    
    return fi