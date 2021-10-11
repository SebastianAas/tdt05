import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
import config
import struct
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class DataProcessor:

    @staticmethod
    def to_var_name(variables, cont=False, hex=False, categorical=False, string=False):
        def intersection(l1, l2):
            res = [value for value in l1 if value in l2]
            return res

        if not cont and not hex and not categorical and not string:
            return variables

        v = []
        if cont:
            v += config.continuous_variables
        if hex:
            v += config.hexadecimal_variables
        if categorical:
            variables += config.categorical_variables
        if string:
            variables += config.string_variables
        return ["f{}".format(x) for x in v]

    @staticmethod
    def normalize(df, columns=None):
        if columns is None:
            return (df - df.min()) / (df.max() - df.min())
        for i in columns:
            df[i] = (df[i] - df[i].min()) / (df[i].max() - df[i].min())
        return df

    def impute(self, df, columns, strategy="mean"):
        imp = SimpleImputer(strategy=strategy, missing_values=np.nan)
        df = imp.fit_transform(df[columns])
        return pd.DataFrame(df, columns)

    @staticmethod
    def hex_to_decimal(df, columns):
        print("columns:", columns)

        def hex_dec(x):
            if type(x) == float:
                return x
            return int("0{}".format(x), 16)

        for i in columns:
            df[i] = df[i].apply(hex_dec)
        return df

    def process(self, data, columns, normalize=False, one_hot_encode=False, impute=True, convert_hex=False,
                save_path="df.csv"):
        print(data.shape)
        if convert_hex:
            data = self.hex_to_decimal(data, self.to_var_name(columns, hex=True))
            print(data.shape)
        if normalize:
            data = self.normalize(data, columns=self.to_var_name(columns, hex=True, cont=True))
            print(data.shape)
        if one_hot_encode:
            data = pd.get_dummies(data, columns=self.to_var_name(columns, categorical=True, string=True))
            print(data.shape)
        if impute:
            data = self.impute(data, columns=self.to_var_name(columns, hex=True, cont=True))
            print(data.shape)
        data.to_csv(save_path)


if __name__ == '__main__':
    data = pd.read_csv("../data/challenge2_test.csv")
    data = data.drop("id", axis=1)
    pr = DataProcessor()
    pr.process(data, columns=config.attributes, normalize=True, convert_hex=True, impute=False,
               save_path="../data/dataframe_test.csv")
