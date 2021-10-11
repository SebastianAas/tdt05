import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

# initialize data
import config
from data_intepretation import DataProcessor

challenge = "../data/challenge2_train.csv"
normalized = "../data/normalized.csv"
one_hot_encoded = "../data/one_hot_encoded.csv"
data = "../data/dataframe.csv"
test = "../data/dataframe_test.csv"

def write_predictions(preds, save_path="../data/predictions.csv"):
    file = open(save_path, "w")
    file.write("id,target\n")

    for id in range(len(preds)):
        file.write(f"{preds[id]},{preds[id][1]}\n")

    file.close()


def data_processing(dataset_path):
    data = pd.read_csv(dataset_path)
    data.drop(columns=data.columns[0], axis=1, inplace=True)
    train = data.fillna(0)
    labels = train[["target"]]
    train = train.drop("target", axis=1)
    return train, labels


train, labels = data_processing("../data/dataframe.csv")
test_data = pd.read_csv(test).fillna(0)
test_data.drop(columns=test_data.columns[0], axis=1, inplace=True)
print("train data: ", train.shape)
print("labels data: ", labels.shape)
print(test_data.head())

x_train, x_val, y_train, y_val = train_test_split(train, labels, test_size=0.2, random_state=0)
cat_features = config.string_variables

cat_features = [x + 1 for x in cat_features]

model = CatBoostClassifier(depth=5,
                           learning_rate=0.01,
                           eval_metric="AUC",
                           early_stopping_rounds=100,
                           loss_function='Logloss',
                           verbose=100)

# train the model
proc = DataProcessor()
c = proc.to_var_name(config.string_variables)
print(c)
model.fit(x_train, y_train, cat_features=c, eval_set=(x_val, y_val))
# make the prediction using the resulting model
preds_class = model.predict(test_data)
preds_proba = model.predict_proba(test_data)


#print("class = ", preds_class)
#print("proba = ", preds_proba)
