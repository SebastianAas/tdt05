import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

# initialize data
import config
from data_processor import DataProcessor

challenge = "../data/challenge2_train.csv"
challenge_test = "../data/challenge2_test.csv"
normalized = "../data/normalized.csv"
one_hot_encoded = "../data/one_hot_encoded.csv"
df_path = "../data/dataframe.csv"
df_test_path = "../data/dataframe_test.csv"


def write_predictions(preds, save_path="../data/predictions.csv"):
    file = open(save_path, "w")
    file.write("id,target\n")

    for id in range(len(preds)):
        file.write(f"{50000+id},{preds[id][1]}\n")

    file.close()


def data_processing(dataset_path, str_columns):
    data = pd.read_csv(dataset_path)
    data.drop(columns=data.columns[0], axis=1, inplace=True)
    labels = data[["target"]]
    train = data.drop("target", axis=1)
    train = data[str_columns].applymap(str)
    return train, labels


def test_data_processing(dataset_path, str_columns):
    data = pd.read_csv(dataset_path)
    data.drop(columns=data.columns[0], axis=1, inplace=True)
    data = data[str_columns].applymap(str)
    return data


cat_features = DataProcessor.df_name(config.attributes)
train, labels = data_processing(challenge, cat_features)
test_data = test_data_processing(challenge_test, cat_features)

x_train, x_val, y_train, y_val = train_test_split(
    train, labels, test_size=0.4, random_state=0
)

model = CatBoostClassifier(
    iterations=700,
    depth=3,
    model_size_reg=3,
    learning_rate=0.1,
    grow_policy="Region",
    score_function="L2",
    l2_leaf_reg=5,
    verbose=True,
    task_type="GPU",
    eval_metric="AUC",
    loss_function="Logloss",
)

# train the model
model.fit(x_train, y_train, cat_features=cat_features, eval_set=(x_val, y_val))
# make the prediction using the resulting model
preds_class = model.predict(test_data)
preds_proba = model.predict_proba(test_data)


# print("class = ", preds_class)
write_predictions(preds_proba)
