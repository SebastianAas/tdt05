import math

import numpy as np
import pandas as pd
import re
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
import config
import struct
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer


class DataProcessor:
    def __init__(self):
        self.cotinuous_cols = config.continuous_variables
        self.hex_cols = config.hexadecimal_variables
        self.categorical_variables = config.categorical_variables

    @staticmethod
    def df_name(variables):
        return ["f{}".format(x) if isinstance(x, int) else x for x in variables ]

    @staticmethod
    def intersection(l1, l2):
        res = [value for value in l1 if value in l2]
        return res

    def get_correct_columns(self, variables: list[int], cont=False, hex=False, categorical=False):

        if cont:
            return self.df_name(self.intersection(variables, self.cotinuous_cols))
        if hex:
            return self.df_name(self.intersection(variables, self.hex_cols))
        if categorical:
            return self.df_name(self.intersection(variables, self.categorical_variables))

    @staticmethod
    def normalize(df, columns=None):
        if columns is None:
            return (df - df.min()) / (df.max() - df.min())
        for i in columns:
            df[i] = (df[i] - df[i].min()) / (df[i].max() - df[i].min())
        return df

    def impute(self, df, columns, strategy="mean"):
        data = df[columns]
        vars = data.columns
        if strategy == "mean":
            imp = SimpleImputer(strategy=strategy, missing_values=np.nan)
            data = imp.fit_transform(data)
        else:
            imp = IterativeImputer(random_state=1, verbose=2)
            data = imp.fit_transform(data)
        df[columns] = pd.DataFrame(data, columns=vars)
        return df

    @staticmethod
    def hex_to_decimal(df, columns):
        def hex_dec(x):
            if type(x) == float:
                return x
            return int("0{}".format(x), 16)

        for i in columns:
            df[i] = df[i].apply(hex_dec)
        return df

    @staticmethod
    def one_hot_encoded_columns(l1, l2):
        res = []
        for value in l2:
            r = re.compile(f"^{value}(_.*|$)")
            t = list(filter(r.match, l1))
            res += t
        return list(dict.fromkeys(res))

    def process(
            self,
        data,
        columns,
        str_columns,
        normalize=False,
        one_hot_encode=False,
        impute=True,
        convert_hex=False,
        save_path="df.csv",
    ):
        if convert_hex:
            data = self.hex_to_decimal(
                data, self.get_correct_columns(columns, hex=True)
            )
            self.cotinuous_cols += self.hex_cols
        if normalize:
            data = self.normalize(
                data, columns=self.get_correct_columns(columns, hex=True)
            )
        if one_hot_encode:
            self.categorical_variables += self.hex_cols
            one_hot_columns = self.get_correct_columns(columns, categorical=True)
            data = pd.get_dummies(
                data, columns= one_hot_columns
            )
            print(data.head())
        if impute:
            data = self.impute(
                data, columns=self.df_name(config.cat_encoded_as_int + config.numerical), strategy="iterative"
            )


        # Transform selected columns into string type
        data[self.df_name(str_columns)] = data[self.df_name(str_columns)].applymap(str)

        columns = self.one_hot_encoded_columns(data.columns.tolist(), self.df_name(columns))

        # Data should only contain the selected columns
        data = data[columns]

        if save_path == None:
            return data

        else:
            data.to_csv(save_path)

def transform(x):
    if x < 1:
        return np.int64(x*10)
    elif math.isnan(x):
        return x
    else:
        return np.int64(x)

if __name__ == "__main__":
    data = pd.read_csv("../data/challenge2_train.csv")
    print(data.head())
    pr = DataProcessor()
    int_encoded = pr.df_name(config.cat_encoded_as_int)
    data[int_encoded] = data[int_encoded].applymap(lambda x: transform(x))
    features = ["id", "target"] + config.attributes
    pr.process(data, columns=features, str_columns=[], normalize=True, convert_hex=True, impute=True,
               save_path="../data/dataframe_train.csv")
    print("Okay")
