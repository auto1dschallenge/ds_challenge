## Task:
Describe how you would improve the model in Question 3 if you had more time.


## Answer

There are mutiple ways that I would like to try to further improve the model submitted.

#### Improved feature generation pipeline
To generate continous features from some categorical variables, e.g. _make_, I take the mean price per category from the entire data set, before splitting the data into test and control. Therefore this feature technically contains information that should not be available to the model during training.
The correct implementation would generate this feature just from the specific training set used for every instance of the model. This is cumbersome to implement given the model evaluation using crossvalidation.
Technically one could argue that given a large enough dataset and a random split the mean price per category should be the same between any train/test split. However, given more time I would definetly make use of sklearns pipeline functionality an build a custom transformer that does the feature transformation on just the training set.

Note: The same issue also applies to the imputation of missing values in the preprocessing of the data.

#### Better feature selection
When selecting the features to use, I only looked at the correlation between individual features and price. I would assume that there exist certain combinations of feature values that strongly influence the price even if the individual features do not. Principal component analysis would be one way to find structures in the feature space that combine interaction between multiple features.

#### Use RandomForest implementation with native support for categorical variables
sklearn's implementation of decision trees does not natively handle categorical variables. Therefore categrocial variables have to be transformed into continous features, either via one-hot encoding or through different ways of feature engineering. I would like to build and train a model using a descision tree implementation that can handle categorical variables directly. Such implementations exist e.g. in R or SparkML.

#### Improved feature imputation for normalized-losses
Given the large number of missing values for normalized-losses, simple ways of imputing the NaN values might not capture sufficient/correct information. An alternative would be to first build a separate model that predicts normalized-losses given car specifications and the symboling, then use that model to impute the missing normalized-losses. Given that normalized-losses seem to have small correlation with the price, this is something I would pursue with low priority.