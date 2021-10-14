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
        return ["f{}".format(x) for x in variables]

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
        print(data.head())
        if strategy == "mean":
            imp = SimpleImputer(strategy=strategy, missing_values=np.nan)
            data = imp.fit_transform(data)
        else:
            imp = IterativeImputer(estimator=ExtraTreesRegressor())
            data = imp.fit_transform(data)
        df[columns] = pd.DataFrame(data, columns)
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
                data, columns=self.get_correct_columns(columns, cont=True), strategy="iterative"
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


if __name__ == "__main__":
    data = pd.read_csv("../data/challenge2_train.csv")
    data = data.drop("id", axis=1)
    pr = DataProcessor()
    pr.process(data, columns=config.attributes, str_columns=[], normalize=True, convert_hex=True, impute=True,
               save_path="../data/dataframe_train.csv")
