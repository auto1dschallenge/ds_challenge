## test lin reg
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from HyperParamTuning import RF_paramSearch

## Define models in the code

def get_lr_base():
    regressor = LinearRegression()
    return regressor

def get_lasso_base():
    regressor = Lasso()
    scaler = StandardScaler()
    return make_pipeline(scaler,regressor)

def get_rf_base():
    regressor = RandomForestRegressor(n_estimators = 20, random_state = 0)
    return regressor

def get_gbt_base():
    regressor = GradientBoostingRegressor(n_estimators = 20, random_state = 0)
    return regressor

def get_rf_prelim():
#    best_params = RF_paramSearch(X,y)
    regressor = RandomForestRegressor(n_estimators = 70,max_depth=5,
                                      random_state = 0)
    return regressor


def get_rf_tuned():
#    best_params = RF_paramSearch(X,y)
    regressor = RandomForestRegressor(n_estimators = 100,
                                      max_depth=10,
                                      max_features="log2",
                                      min_samples_leaf=1,
                                      min_samples_split = 5,
                                      criterion = "mse",
                                      bootstrap = True,
                                      random_state = 0)
    return regressor