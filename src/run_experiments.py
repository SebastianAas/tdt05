import os
from datetime import datetime
from pathlib import Path
from typing import Union
import tensorflow as tf

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# initialize data
import config as cf
from data_processor import DataProcessor


class ExperimentConfig:
    def __init__(
            self,
            number_of_variables,
            cat_features: list[int],
            cross_validation: int = 10,
            normalize: bool = False,
            convert_hex: bool = False,
            one_hot_encoded: bool = False,
            impute: bool = False,
    ):
        self.number_of_variables = number_of_variables
        self.cat_features = cat_features
        self.cv = cross_validation
        self.normalize = normalize
        self.convert_hex = convert_hex
        self.one_hot_encoded = one_hot_encoded
        self.impute = impute


class GridConfig:
    def __init__(
            self, depth: list[int], learning_rate: list[float],
            l2_leaf_reg: list[int], grow_policy: list[str], border_count: list[int]
    ):
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.grow_policy = grow_policy
        self.border_count = border_count


class CatConfig:
    def __init__(
            self, depth: int, learning_rate: float, l2_leaf_reg: int,
    ):
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg


class NNConfig:
    def __init__(self, learning_rate, layers):
        self.learning_rate = learning_rate
        self.layers = layers


class Experiment:
    def __init__(self, dataset_path: str, testset_path: str, experiment_config: ExperimentConfig,
                 model_config: Union[GridConfig, CatConfig, NNConfig]):
        self.experiment_config = experiment_config
        self.model_config = model_config
        self.dp = DataProcessor()
        self.data = pd.read_csv(dataset_path)
        self.test = pd.read_csv(testset_path)

    def run_experiments(self):
        variables_with_most_uniques = self.data.iloc[:, 2:].nunique().sort_values().index.to_list()
        for i in self.experiment_config.number_of_variables:
            print("Number of variables: ", i, "\n")
            columns = [int(x.replace("f", "")) for x in variables_with_most_uniques[:i]]
            str_columns = [] if self.experiment_config.one_hot_encoded else self.experiment_config.cat_features
            labels = self.data["target"]
            data = self.data.drop("target", axis=1)
            traintest = self.data.append(self.test)
            traintest = self.dp.process(
                traintest,
                columns=columns,
                str_columns=str_columns,
                normalize=self.experiment_config.normalize,
                convert_hex=self.experiment_config.convert_hex,
                one_hot_encode=self.experiment_config.one_hot_encoded,
                impute=self.experiment_config.impute,
                save_path=None
            )
            train = traintest[:len(self.data)]
            test_data = traintest[len(self.data):]
            """
            test_data = self.dp.process(
                self.test,
                columns=columns,
                test=True,
                str_columns=str_columns,
                normalize=self.experiment_config.normalize,
                convert_hex=self.experiment_config.convert_hex,
                one_hot_encode=self.experiment_config.one_hot_encoded,
                impute=self.experiment_config.impute,
                save_path=None
            )
            """
            x_train, x_val, y_train, y_val = train_test_split(
                train, labels, test_size=0.02
            )
            if type(self.model_config) == GridConfig:
                self.run_grid_search(train, labels, test_data, self.dp.df_name(self.dp.intersection(self.experiment_config.cat_features, columns)))
            elif type(self.model_config) == NNConfig:
                model = self.train_nn_model(train, labels, test_data)
            else:
                self.run_single_experiment(train, labels, test_data)

    def run_grid_search(self, train, labels, test_data, cat_features):
        model = CatBoostClassifier(
            early_stopping_rounds=100,
            eval_metric="AUC",
            loss_function="Logloss",
            cat_features=cat_features,
            verbose=100,
        )
        model.grid_search(vars(self.model_config), train, y=labels)
        predictions = model.predict_proba(test_data)
        accuracy = (model.get_best_iteration(), model.get_best_score())
        self.write_results(predictions, accuracy, len(train.columns))

    def run_grid_search_cv(self, train, labels, test_data):
        model = CatBoostClassifier(
            early_stopping_rounds=100,
            eval_metric="AUC",
            cat_features=self.experiment_config.cat_features,
        )

        gscv = GridSearchCV(estimator=model, param_grid=vars(self.model_config), scoring="AUC", cv=10)
        gscv.fit(train, y=labels)
        predictions = gscv.predict_proba(test_data)
        print(gscv.best_params_)
        print(gscv.best_estimator_)
        accuracy = (model.get_best_iteration(), model.get_best_score())
        self.write_results(predictions, accuracy, len(train.columns))

    def train_nn_model(self, train, labels, test_data):
        x_train, x_val, y_train, y_val = train_test_split(
            train, labels, test_size=0.2
        )
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(len(train.columns), activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

        predictions = model.predict(test_data)
        predictions = [(1 - x, x) for x in predictions]
        accuracy = model.evaluate(test_data)
        print(accuracy)
        print(predictions)
        self.write_results(predictions, accuracy, len(train.columns))
        return model

    def run_single_experiment(
            self,
            train, labels, test_data

    ):
        model = CatBoostClassifier(
            learning_rate=model_config.learning_rate,
            depth=model_config.depth,
            l2_leaf_reg=model_config.l2_leaf_reg,
            grow_policy="Lossguide",
            eval_metric="AUC",
            loss_function="Logloss",
            border_count=254,
            verbose=100,
            early_stopping_rounds=100,
        )
        x_train, x_val, y_train, y_val = train_test_split(
            train, labels, test_size=0.01
        )

        if self.experiment_config.one_hot_encoded:
            cat_features = []
        else:
            columns = [int(x.replace("f", "")) for x in train.columns.tolist()]
            cat_features = self.dp.df_name(self.dp.intersection(columns, self.experiment_config.cat_features))

        model.fit(
            X=x_train,
            y=y_train,
            cat_features=cat_features,
            eval_set=(x_val, y_val),
        )
        predictions = model.predict_proba(test_data)
        accuracy = (model.get_best_iteration(), model.get_best_score())
        self.write_results(predictions, accuracy, len(train.columns))

    def load_data(self):
        pass

    def preprocess_data(self, dataset_path, str_columns, test):
        data = pd.read_csv(dataset_path)
        data.drop(columns=data.columns[0], axis=1, inplace=True)
        if test:
            data = data[str_columns].applymap(str)
            return data
        else:
            labels = data[["target"]]
            train = data.drop("target", axis=1)
            train = data[str_columns].applymap(str)
            return train, labels

    def train(self):
        pass

    def write_results(
            self, predictions, accuracy, number_of_variables
    ):
        AUC = accuracy[1]["validation"]["AUC"]
        logloss = accuracy[1]["validation"]["Logloss"]
        score_file = Path("../experiments/score_board.csv")
        if not score_file.is_file():
            with ("../experiments/score_board.csv", "w+") as f:
                f.write(",AUC,Logloss,file_path")
        score_board = pd.read_csv("../experiments/score_board.csv")
        dir_path = f"../experiments/{number_of_variables}"
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        Path(f"{dir_path}/{time}").mkdir(exist_ok=True, parents=True)
        file_path = f"{time}/predictions.csv"
        save_path = f"{dir_path}/{file_path}"
        results_path = f"{dir_path}/{time}/results.txt"
        log = [{"AUC": AUC, "Logloss": logloss, "file_path": results_path}]
        d = pd.DataFrame(log)
        score_board = score_board.append(d).sort_values(by=["AUC"], ascending=False)
        score_board.to_csv("../experiments/score_board.csv", index=False)
        self.write_predictions(predictions, save_path=save_path)

        with open(results_path, "w") as f:
            f.write("## Config\n")
            f.write(str(vars(self.experiment_config)))
            f.write("\n")
            f.write("## Results\n")
            f.write(f"Best iteration: {accuracy[0]}\n")
            f.write(f"Best accuracy: {accuracy[1]}\n")

    def write_predictions(self, preds, save_path):
        file = open(save_path, "w")
        file.write("id,target\n")

        for id in range(len(preds)):
            file.write(f"{50000 + id},{preds[id][1]}\n")

        file.close()


if __name__ == "__main__":
    dataset_path = "../data/challenge2_train.csv"
    testset_path = "../data/challenge2_test.csv"
    cat_config = GridConfig(
        learning_rate=cf.learning_rate, depth=cf.depths, l2_leaf_reg=cf.l2_leaf_reg, grow_policy=cf.growth_strategy,
        border_count=cf.border_count
    )
    model_config = CatConfig(
        learning_rate=0.0811, depth=3, l2_leaf_reg=2
    )
    nn_config = NNConfig(learning_rate=0.1, layers=[512, 256, 128])
    cat_features = cf.attributes
    ex_config = ExperimentConfig(
        number_of_variables=cf.number_of_variables,
        cat_features=cat_features,
        convert_hex=False,
        normalize=False,
        one_hot_encoded=False,
        impute=False
    )
    ex = Experiment(dataset_path, testset_path, ex_config, model_config)
    ex.run_experiments()
