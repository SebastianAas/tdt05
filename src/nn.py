import tensorflow as tf
import numpy as np
import pandas as pd
import config as cf
from tensorflow import keras
from tensorflow.keras import layers
from data_processor import DataProcessor
from src.run_experiments import ExperimentConfig

dataframe = pd.read_csv("../data/challenge2_train.csv")
test = pd.read_csv("../data/challenge2_test.csv")


cat_features = cf.categorical_variables + cf.hexadecimal_variables

experiment_config = ExperimentConfig(
    number_of_variables=cf.number_of_variables,
    cat_features=cat_features,
    convert_hex=False,
    normalize=False,
    one_hot_encoded=True,
)

variables_with_most_uniques = dataframe.iloc[:, 2:].nunique().sort_values().index.to_list()
columns = [int(x.replace("f", "")) for x in variables_with_most_uniques[:26]]
columns.remove(5)
str_columns = []
labels = dataframe["target"]
data = dataframe.drop("target", axis=1)
traintest = dataframe.append(test)
traintest = DataProcessor().process(
    traintest,
    columns=columns,
    str_columns=str_columns,
    normalize=experiment_config.normalize,
    convert_hex=experiment_config.convert_hex,
    one_hot_encode=experiment_config.one_hot_encoded,
    impute=experiment_config.impute,
    save_path=None
)

train = traintest[:len(dataframe)]
test_data = traintest[len(dataframe):]

train.insert(0, "target", labels)

val_dataframe = train.sample(frac=0.2, random_state=1337)
train_dataframe = train.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train)
val_ds = dataframe_to_dataset(val_dataframe)

for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)


