## Task:
Implement the model you described in question 2 in R or Python. The code has to retrieve the data, train and test a statistical model, and report relevant performance criteria. 



## Answer

### Build a statistical model to predict car prices

A jupyter notebook used for explorary data analysis can be found in _./../DataExploration/_.
The code used to build the actual model lies in _./../python/_. 
* The code can be executed by running _main.py_.

This retrieves the dataset, prepares the data for training (NaN handling, feature engineering, encoding and selection), then trains and evaluates the model. After running, the code reports error metrics for the model in the form of the RMSE. For specifics on the model evaluation, please see the response to question 4.

* _Preprocessing.py_ contains classes for loading the data as well as building and encoding features.
* _eval.py_ contains functions for model evaluation.
* _feature_selection.py_ contains functions for for feature selection, i.e. dropping features without predictive power.
* _models.py_ contains the various different models tried and used in the modelling process.
* _HyperParamTuning.py_ contains functions for finding optimal parameter settings for a RandomForest.