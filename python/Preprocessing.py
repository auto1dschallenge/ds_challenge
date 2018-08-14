import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from collections import Counter
from sklearn.preprocessing import StandardScaler


class DataPreparation:
    ## Class to load and prepare the dataset
    def __init__(self):
        self.data = pd.read_csv('./../Auto1-DS-TestData.csv',na_values = "?")
        self.train_data = []
        self.validation_data = []
    
    def rm_na(self,col):
        # removes all columns with null entries in the specified col
        self.data= self.data[self.data[col].notnull()]

    
        
    
    def impute_losses(self,method):
        ## Imputes normalized-losses
        print(method)
        if method != "median_by_symbol":
            print("Mean Imputer for losses")
            imp = Imputer(strategy = method)
            tmp = imp.fit_transform(self.data["normalized-losses"].reshape(-1, 1))
            self.data["normalized-losses"] = tmp
        else:
            print("Median by Symbol Imputer for losses")
            medians=self.data[["symboling","normalized-losses"]].groupby("symboling",as_index=False).median()
            for index,row in medians.iterrows():
                self.data.loc[
                        (self.data["normalized-losses"].isnull()) &
                        (self.data["symboling"]==row["symboling"]),
                        ["normalized-losses"]] = row["normalized-losses"]
        
    def impute_numerical(self,col,method = "mean"):
        # impute values in numerical column specified by col
        imp = Imputer(strategy = method)
        tmp = imp.fit_transform(self.data[col].reshape(-1, 1))
        self.data[col] = tmp
        
    def impute_categorical(self,col):
        # imputes categorical column col with most common entry
        c = Counter(col)
        mc = c.most_common(1)[0][0]
        self.data[col] = self.data[col].fillna(mc)
        
    def impute_nans(self,method_num = "mean" ,method_losses = "median_by_symbol"):
        # if called, loops through all columns and imputes nan values
        tmpcols = self.data.columns.delete(self.data.columns.get_loc("price"))
        for c in tmpcols:
            if str(self.data[c].dtype) == "object":
                if self.data[c].isnull().sum() > 0:
                    print("Imputing " + c)
                    self.impute_categorical(c)
            elif c == "normalized-losses":
                    print("Imputing " + c)                
                    self.impute_losses(method_losses)
            else:
                if self.data[c].isnull().sum() > 0:
                    print("Imputing " + c)
                    self.impute_numerical(c,method_num)
                    
    def find_nan(self):
        # Find all NaN values in dataset
        for i in self.data.columns:
            if self.data[i].isnull().sum() > 0:
                print(str(self.data[i].isnull().sum()) + "\t missing entries in " + i) 
                
    def replace_cat_variable(self,cols):
        # Replaces specified categorical columns cols with numerical values where category is replaced
        # with the mean price per category
        for i in cols:
            print("Replace " + i + " with mean price value by category")
            medians=self.data[[i,"price"]].groupby(i,as_index=False).mean()
            for index,row in medians.iterrows():
                self.data.loc[
                        self.data[i]==row[i],i] = row["price"]
                
    def cluster_groups(self):
        # Transforms variables drive-wheels and fuel-system into binary categorical variables
        cluster_dict = { "drive-wheels": {"rwd": "main", "fwd": "other","4wd": "other" },
                        "fuel-system": {'mpfi': 'main', '2bbl': 'other','mfi': 'other'
                                        ,'1bbl': 'other','spfi': 'other','4bbl': 'other'
                                        ,'idl': 'other','spdi': 'other'} }
        self.data.replace(cluster_dict,inplace=True)



class FeatureEncoding:
    # class do encode categorical variables
        def __init__(self,df):
            self.data = df
            self.transformed_data = []
        
        def encode_binary_cols(self,cols):
            # Encode binary categorical variables using LabelEncoder
            for i in cols:
                lbe = LabelEncoder()
                self.data[i] = lbe.fit_transform(self.data[i])

        def encode_cat_cols(self,cols):
            # For non-binary categorical variables uses LabelEncoder and one-hot encoding to create
            # n-1 continous variables where n is the number of categories
            for i in cols:
                lbe = LabelEncoder()
                self.data[i] = lbe.fit_transform(self.data[i])
 
            self.data = pd.get_dummies(self.data,columns = cols)
            keep_cols = [c for c in self.data.columns if c.lower()[-2:] != '_0']
            self.data = self.data[keep_cols]
            
        def encode_cylinders(self):
            # transforms cylinder variables into a continous numerical variable
            cylinder_dict = { "num-of-cylinders": {"four": 4, "six": 6, "five": 5, "eight": 8,
                                  "two": 2, "twelve": 12, "three":3 }}
            self.data.replace(cylinder_dict,inplace=True)
                
            

def get_scaled_data(X,y):
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    
    Xs = sc_x.fit_transform(X)
    ys = sc_y.fit_transform(y)
    
    return Xs,ys
    