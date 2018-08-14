## Task:
Explain each and every of your design choices (e.g., preprocessing, model selection, hyper parameters, evaluation criteria). Compare and contrast your choices with alternative methodologies. 


## Answer

All the points within the modelling pipeline can broadly be categorized into these topics:

* Data Preprocessing
* Choice of Evaluation Metrics
* Model Selection
* Feature Selection and Engineering
* Model Tuning / Hyper parameter settings
* Results


### Data Preprocessing

* Since the goal of the model presented is to predict the price, data preprocessing starts with removing those 4 rows of the dataset that are missing price information. 
* There are a couple of columns with 2-4 missing values for which mean imputation is used.
* The _normalized-losses_ column contains 41 missing values (with total 201 entries in the dataset). Therefore I assume simple mean imputation will not be sufficient to keep this columns relevance to price prediction. Given the explanation of both the _normalized-losses_ and _symboling_ I assume a correlation between both values which is confirmed by exploratory data analysys. Thus I replace the NaN values in the _normalized-losses_ with the median _normalized-losses_ per symboling rating.

### Choice of Evaluation Metric
There are two questions concerning model evaluation:
1. Which metric to use?
2. On which dataset to evaluate the metric?

* For the first question, I choose RMSE as the metric to evaluate. The RMSE has the advantage of punishing larger errors more severely, helping to keep the model accurate over the entire price range as errors with the same relative magnitude are punished more strongly for higher prices. In contrast MAE would put more emphasis on accuracy for the lower price range. Also, the RMSE values are on the same scale as the target variable, making them easier to interpret as e.g. a relative metric like the r^2 score.
* The default approach for evaluating the model would be to split the data into a train and validation set, then evaluate the model on the validation set exclusively. However given the small size of the dataset the outcome of this process depends very strongly on the individual validation set. To define a more stable error metric, in addition to the RMSE on the validation set, I will evalute the model on the training set via 5-fold crossvalidation. The errors on each fold are then averaged to define a models performance on the training set.

To summarize, the model is evaluated in two ways:
1. 5-fold crossvalidation on the training set.
2. The model is then trained on the full training set and evaluated on the validation set exactly once.

### Model Selection
Since the model contains a _large_ (compared to the number of entries in the dataset) number of both continous and categorical variables I train a RandomForest model on the data. Desicion Trees can  handle nonlinearites in the data and are good at dealing with a large number of features. The ensembling helps to prevent overfitting, something which i am worried about given the small size of the training data. Linear regression models without some form of regularization did not perform well on the dataset as the results where mostly overfit and even with regularization generalized badly to the test data.

### Feature Selection and Engineering

#### Encoding of categorical variables
Since sklearn's implementation of RandomForest does not support categorical variables, all categorical variables are transformed into continous variables by the following process:

* _num-of-cylinders_ is directly translated into a continous variable
* all categorical variables are encoded in numerical variables using sklearn's LabelEncoder
* One-Hot encoding is used to to convert all non-binary categorical variables into n-1 numerical variables (where n is the number of levels of the variable)

This process results in a large number of individual variables (~50). In addition, some categorical variables, e.g. _make_, have a lot of levels. Through the process of one-hot encoding these get split into many different features where every individual feature has little information value and will not be considered as strongly when splits the decision trees are determined.

#### Feature Selection and Engineering
To reduce the number of variables in the model, 2 approaches are followed. 
1. For the continous variables, only consider those features whose absolute correlation coefficient with _price_ is > 0.5
2. For the non-binary categorical variables either:
  * transform them into binary classes based on distribution of price among the categories (_drive-wheels,fuel-system_)
  * transform into one continous variable by assigning each entry the mean _price_ value within the category (_make,body-style and engine-type_)

This way we end up with a total of 16 different features to use in the RandomForest model.

This process of feature selection is very manual and involves some subjective decisions and arbitrary cutoff points. For a more automated way of feature selection, I tried Lasso Regression. Lasso Regression penalizes the absolute value of the regression coefficents and as such tries to minimize the amount of non-zero coefficents used in the model (and as such the features taken into consideration for the prediction). This way it can be used to automatically select features to use for training a model. Using the automatic Lasso approach to feature selection performed worse for both the  full feature set and the reduced set after feature engineering.

#### Model Tuning / Hyper parameter settings
To set the final parameters of the RandomForest model, I started with a base learner (20 trees and no max_depth) and used grid search (GridSearchCV) and random parameter search (RandomSearchCV) with crossvalidation to find optimal hyperparameters for the RandomForest on the training set.


#### Results
After these steps, I assess the model performance using the above mentioned evaluation metrics.
To asses which model performed _best_, I only look at the model performance using crossvalidation on the training set. The _best_ model is then exposed to the validation set exactly once.
The best performing RandomForest model reaches an RMSE of _2079.69_ on the training set using 5-fold crossvalidation and an RMSE of _2147.53_ on the validation set.