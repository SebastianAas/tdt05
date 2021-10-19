import tensorflow as tf
import numpy as np
import pandas as pd
import config as cf
from tensorflow import keras
from tensorflow.keras import layers
from data_processor import DataProcessor
from src.run_experiments import ExperimentConfig
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup

dataframe2 = pd.read_csv("../data/challenge2_train.csv")
test = pd.read_csv("../data/challenge2_test.csv")
dataframe = pd.read_csv("../data/dataframe_train.csv")

def get_name(x):
    return DataProcessor.df_name(x)

cat_features = cf.cat_endcoded_as_string

experiment_config = ExperimentConfig(
    number_of_variables=cf.number_of_variables,
    cat_features=cat_features,
    convert_hex=False,
    normalize=False,
    one_hot_encoded=False,
)

variables_with_most_uniques = dataframe.iloc[:, 3:].nunique().sort_values().index.to_list()
columns = [int(x.replace("f", "")) for x in variables_with_most_uniques[:23]]
dataframe[get_name(cf.cat_encoded_as_int)] = dataframe[get_name(cf.cat_encoded_as_int)].applymap(int)
str_columns = cf.cat_endcoded_as_string
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

print(traintest.dtypes)

train = traintest[:len(dataframe)]
test_data = traintest[len(dataframe):]

train.insert(0, "target", labels)

val_dataframe = train.sample(frac=0.3, random_state=1337)
train_dataframe = train.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

print(train.head())
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature



def intersection(l1, l2):
    res = [value for value in l1 if value in l2]
    return res


# Categorical features encoded as integers
cat_as_int = []
for cat in get_name(intersection(columns, cf.cat_encoded_as_int)):
    cat_as_int.append((cat, keras.Input(shape=(1,), name=cat, dtype="int64")))

# Categorical feature encoded as string
cat_as_string = []
for feat in get_name(intersection(columns,cf.cat_endcoded_as_string)):
    cat_as_string.append((feat, keras.Input(shape=(1,), name=feat, dtype="string")))

# Numerical features
numerical = []
for feat in get_name(intersection(columns,cf.numerical)):
    numerical.append((feat, keras.Input(shape=(1,), name=feat)))

all_inputs = cat_as_int + cat_as_string + numerical


# Integer categorical features
cat_int_encoded = []
for (name, input) in cat_as_int:
    cat_int_encoded.append(encode_categorical_feature(input, name, train_ds, False))

# String categorical features
cat_string_encoded = []
for (name, input) in cat_as_string:
    cat_string_encoded.append(encode_categorical_feature(input, name, train_ds, True))

# Numerical features
num_encoded = []
for (name, input) in numerical:
    num_encoded.append(encode_numerical_feature(input, name, train_ds))

features = cat_int_encoded + cat_string_encoded + num_encoded
all_features = layers.concatenate(features)

inputs = [x[1] for x in all_inputs]

def write_predictions(preds, save_path):
    file = open(save_path, "w")
    file.write("id,target\n")

    for id in range(len(preds)):
        file.write(f"{50000 + id},{preds[id][0][0]}\n")

    file.close()

x = layers.Dense(32, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
x = layers.Dense(56, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(10, activation="relu")(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.fit(train_ds, epochs=10, validation_data=val_ds)
test_ds = test_data.to_dict("records")
input_dict = [{name: tf.convert_to_tensor([value]) for name, value in row.items()} for row in test_ds]
predictions = []
i = 0
print("Starting to predict")
predictions = model.predict(input_dict)
"""
for input in input_dict:
    print("i: ", i)
    i += 1
    predictions.append(model.predict(input))
"""
write_predictions(predictions, "../data/predictions_nn.csv")
