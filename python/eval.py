from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
import numpy as np

def evalCV(pred,X,y,folds=5):
    # Evaluate a predictor pred on the dataset X with labels y using
    # 5 fold crossvalidation 
    # Returns average RMSE across all folds
    scores = cross_validate(pred,X,y,cv=folds,scoring="neg_mean_squared_error")
    rmse = np.sqrt(np.abs(scores["test_score"])).mean()
    return rmse

def eval_val(pred,X,y):
    # Returns RMSE of model pred on dataset X with labels y
    mse = mean_squared_error(pred.predict(X),y)
    return np.sqrt(mse)