import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import Preprocessing as pp
import feature_selection as sel
import eval
import models
from HyperParamTuning import RF_paramSearch

# Read data
dp = pp.DataPreparation()
dp.find_nan() ## Check the number of missing values
dp.rm_na("price") ## Remove columns with missing price values
dp.impute_nans() ## Impute nan values according to documention
dp.find_nan() ## Check if impute was succesful

dp.replace_cat_variable(["body-style","make","engine-type"]) ## Transform some categorical variables into continous variables
dp.cluster_groups() ## Transform some non-binary variables into binary categorical variables


raw_data = dp.data ## Save data set after transformation
bin_cols = ["fuel-type","aspiration","num-of-doors","engine-location",'drive-wheels','fuel-system'] 
cat_cols = []
#cyl_col = "num-of-cylinders"


fe = pp.FeatureEncoding(raw_data) 
fe.encode_cylinders() ## transform cylinder variable into continous variable
fe.encode_binary_cols(bin_cols) ## Encode binary categorical columns using LabelEncoder
#fe.encode_cat_cols(cat_cols)

data = fe.data
y = data["price"].reshape(-1,1)  ## Separate features and labels
X = data.drop("price",axis=1)


# Feature Selection

new_cols = sel.correlation_cols(X) 
#new_cols = lasso_coefficients(X,y)

X = X[new_cols] ## select features used as descibed in the documentation


## Split Data
## Define Test and Train Set

X_train, X_validation,y_train,y_validation = train_test_split(
                X,y, test_size = 0.2, random_state = 0)


## Get Model

model  = models.get_rf_tuned() ## Get tuned RandomForest model


# Train and eval model on Training Set

rmse_cv = eval.evalCV(model,X_train,y_train.ravel(),5)


# Train Model on entire training data

model.fit(X_train,y_train)


# Eval model on Validation set

rmse_val = eval.eval_val(model,X_validation,y_validation)


# Eval model on test set

rmse_train = eval.eval_val(model,X_train,y_train)


## Print Results

plt.scatter(y_train,model.predict(X_train))
plt.title("Predicted vs Observed Error on Training Set")
plt.show()

plt.scatter(y_train,(model.predict(X_train)-y_train.ravel())/y_train.ravel())
plt.title("Percentage Error on Training Set")
plt.show()


print("")
print("Model RMSE \nevaluated on full training data:\n" + str(rmse_train))
print("")


plt.scatter(y_validation,model.predict(X_validation))
plt.title("Predicted vs Observed Error on Validation Set")
plt.show()

plt.scatter(y_validation,(model.predict(X_validation)-y_validation.ravel())/y_validation.ravel())
plt.title("Percentage Error on Validation Set")
plt.show()

print("")
print("Model RMSE \nevaluated on training data via crossvalidation:\n" + str(rmse_cv))
print("")
print("Model RMSE \nevaluated on validation set:\n" + str(rmse_val))
print("")