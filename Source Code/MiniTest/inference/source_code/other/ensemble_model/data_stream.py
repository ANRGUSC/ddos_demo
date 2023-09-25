import collections
import sys
import glob
import pylab as pl
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    multilabel_confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
import random
import os
from multiprocessing import Pool
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pickle import load
from scipy.stats import norm, cauchy
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("INFO")

sys.path.append("../")
sys.path.append("../../")
sys.path.append("./source_code/")

import project_config as CONFIG


def prepare_output_directory(output_path):
    """Prepare the output directory by deleting the old files and create an empty directory.

    Keyword arguments:
    output_path -- path to the output directory
    """
    dir_name = str(os.path.dirname(output_path))
    # os.system("rm -rf " + dir_name)
    os.system("mkdir -p " + dir_name)


def load_dataset(path):
    """Load the dataset

    Keyword arguments:
    path -- path to the dataset csv file
    """
    data = pd.read_csv(path)
    data["TIME"] = pd.to_datetime(data["TIME"])
    return data


def LSTM_MM_WC_get_input_target(data, num_labels, time_window, scaler):
    """
    Generate the training dataset for a given time window
    Args:
        data: The original dataset containing the training features and labels
        num_labels: Number of labels
        time_window: The time window for generating the training dataset
        scaler_save_path: The path for storing the scaler function

    Returns:
        X_out: The training dataset input
        y_out: The training dataset label
        scaler: The scaler function for the input dataset
    """
    temp_columns = []
    for column in data.columns:
        if "PACKET_" in column:
            temp_columns.append(column)
    temp_columns.append("ATTACKED")
    X_out = []
    y_out = []

    temp = data[temp_columns]
    df_out = pd.DataFrame()
    X = temp.iloc[:, 0:-num_labels]
    y = temp.iloc[:, -num_labels:]
    X = np.asarray(X).astype(float)
    y = np.asarray(y).astype(float)
    X = scaler.transform(X)

    df_out = df_out.append(data.iloc[time_window - 1 :, :]).reset_index(drop=True)
    for i in range(X.shape[0] - time_window + 1):
        X_out.append(X[i : i + time_window])
        y_out.append(y[i + time_window - 1])

    X_out, y_out = np.array(X_out), np.array(y_out)
    return X_out, y_out, df_out


def LSTM_MM_NC_get_input_target(data, num_labels, time_window, scaler):
    """
    Generate the training dataset for a given time window
    Args:
        data: The original dataset containing the training features and labels
        num_labels: Number of labels
        time_window: The time window for generating the training dataset
        scaler_save_path: The path for storing the scaler function

    Returns:
        X_out: The training dataset input
        y_out: The training dataset label
        scaler: The scaler function for the input dataset
    """
    temp_columns = ["PACKET", "ATTACKED"]
    X_out = []
    y_out = []

    temp = data[temp_columns]
    df_out = pd.DataFrame()
    X = temp.iloc[:, 0:-num_labels]
    y = temp.iloc[:, -num_labels:]
    X = np.asarray(X).astype(float)
    y = np.asarray(y).astype(float)
    X = scaler.transform(X)

    df_out = df_out.append(data.iloc[time_window - 1 :, :]).reset_index(drop=True)
    for i in range(X.shape[0] - time_window + 1):
        X_out.append(X[i : i + time_window])
        y_out.append(y[i + time_window - 1])

    X_out, y_out = np.array(X_out), np.array(y_out)
    return X_out, y_out, df_out


def MM_load_model_weights(model_path_input, node, metric, mode):
    """
    Load and return the neural network model weight
    Args:
        model_path_input: path to the model
        node: the node ID for loading the model
        metric: the metric to use for loading the model
        mode: the mode of the metric to use for loading the model

    Returns:
        model: The neural network model with loaded weights
        scaler: The scaler function for the input dataset
    """
    saved_model_path = model_path_input + str(node) + "/"
    scaler_path = saved_model_path + "scaler.pkl"
    model_path = saved_model_path + "final_model/"
    logs_path = saved_model_path + "logs/logs.csv"
    logs = pd.read_csv(logs_path)
    metrics = list(logs.columns)
    metrics.remove("epoch")

    if mode == "max":
        logs = logs.sort_values(by=[metric], ascending=False).reset_index(drop=True)
    elif mode == "min":
        logs = logs.sort_values(by=[metric]).reset_index(drop=True)

    epoch = str((int)(logs["epoch"][0]) + 1).zfill(4)
    checkpoint_path = saved_model_path + "checkpoints/all/weights-" + epoch

    model = tf.keras.models.load_model(model_path)
    model.load_weights(checkpoint_path)
    # model.summary()
    scaler = load(open(scaler_path, "rb"))

    return model, scaler


def main():
    seed = 1
    tf.random.set_seed(seed)
    random.seed(seed)

    metric = "val_binary_accuracy"
    mode = "max"
    k_list = np.array([0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1])
    k_list = np.round(k_list, 2)
    time_window = 10
    num_labels = 1
    upsample = "without"
    model_architecture = "multiple_models_with_correlation"

    LSTM_MM_WC_model_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training_lstm/current_features_aggregate_all_k/Output_"
        + upsample
        + "_upsample/"
        + model_architecture
        + "/saved_model/"
    )

    upsample = "without"
    model_architecture = "multiple_models_without_correlation"

    LSTM_MM_NC_model_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training_lstm/current_features_aggregate_all_k/Output_"
        + upsample
        + "_upsample/"
        + model_architecture
        + "/saved_model/"
    )

    test_dataset_path = (
        CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/test_data/test_data.csv"
    )
    test_dataset = load_dataset(test_dataset_path)

    output_path = "./Output/Data_Stream/data_stream_"
    prepare_output_directory(output_path)
    df = pd.DataFrame()
    attack_ratios = test_dataset["ATTACK_RATIO"].unique()
    attack_durations = test_dataset["ATTACK_DURATION"].unique()
    nodes = test_dataset["NODE"].unique()
    k_list = test_dataset["ATTACK_PARAMETER"].unique()
    for node in nodes:
        print("****************************************************")
        print("********************  Node = ", node, "***********************")
        print(datetime.now())
        print("****************************************************")

        LSTM_MM_WC_model, LSTM_MM_WC_scaler = MM_load_model_weights(
            LSTM_MM_WC_model_path, node, metric, mode
        )
        LSTM_MM_NC_model, LSTM_MM_NC_scaler = MM_load_model_weights(
            LSTM_MM_NC_model_path, node, metric, mode
        )

        data_1 = test_dataset.loc[test_dataset["NODE"] == node]
        for k in k_list:
            data_2 = data_1.loc[data_1["ATTACK_PARAMETER"] == k]
            for attack_ratio in attack_ratios:
                data_3 = data_2.loc[data_2["ATTACK_RATIO"] == attack_ratio]
                for attack_duration in attack_durations:
                    data_4 = data_3.loc[data_3["ATTACK_DURATION"] == attack_duration]
                    if data_4.shape[0] == 0:
                        continue
                    data_4 = data_4.sort_values(by=["TIME"]).reset_index(drop=True)

                    (
                        X_LSTM_MM_WC,
                        y_LSTM_MM_WC,
                        df_LSTM_MM_WC,
                    ) = LSTM_MM_WC_get_input_target(
                        data_4, num_labels, time_window, LSTM_MM_WC_scaler
                    )
                    (
                        X_LSTM_MM_NC,
                        y_LSTM_MM_NC,
                        df_LSTM_MM_NC,
                    ) = LSTM_MM_NC_get_input_target(
                        data_4, num_labels, time_window, LSTM_MM_NC_scaler
                    )

                    data_4 = df_LSTM_MM_WC
                    # init_pred = np.rint(LSTM_MM_WC_model.predict(X_LSTM_MM_WC))
                    pred_LSTM_MM_WC = np.rint(LSTM_MM_WC_model.predict(X_LSTM_MM_WC))
                    pred_LSTM_MM_WC = np.reshape(
                        pred_LSTM_MM_WC, pred_LSTM_MM_WC.shape[0]
                    )
                    pred_LSTM_MM_NC = np.rint(LSTM_MM_NC_model.predict(X_LSTM_MM_NC))
                    pred_LSTM_MM_NC = np.reshape(
                        pred_LSTM_MM_NC, pred_LSTM_MM_NC.shape[0]
                    )
                    init_pred = pred_LSTM_MM_WC

                    benign_que = collections.deque(maxlen=100)
                    attack_que = collections.deque(maxlen=100)

                    for index, row in data_4.iterrows():
                        if data_4.loc[index, "PACKET"] > 0 and init_pred[index] == 0:
                            benign_que.append(data_4.loc[index, "PACKET"])
                        elif data_4.loc[index, "PACKET"] > 0 and init_pred[index] == 1:
                            attack_que.append(data_4.loc[index, "PACKET"])

                        if len(benign_que) > 0 and len(attack_que) > 0:
                            benign_parameter = cauchy.fit(list(benign_que))
                            attack_parameter = cauchy.fit(list(attack_que))

                            k_estimate = max(
                                attack_parameter[0] / benign_parameter[0] - 1, 0
                            )
                            if k_estimate < 0.3:
                                pred = pred_LSTM_MM_WC[index]
                            else:
                                pred = pred_LSTM_MM_NC[index]
                        else:
                            k_estimate = 0
                            pred = pred_LSTM_MM_WC[index]

                        data_4.loc[index, "pred_LSTM_MM_WC"] = pred_LSTM_MM_WC[index]
                        data_4.loc[index, "pred_LSTM_MM_NC"] = pred_LSTM_MM_NC[index]
                        data_4.loc[index, "k_estimate"] = k_estimate
                        data_4.loc[index, "pred"] = pred

                    df = df.append(data_4)
            output_path = (
                "./Output/Data_Stream/data_stream_" + str(datetime.now()) + ".csv"
            )
            df.to_csv(output_path, index=False)
            # df = df[['k_estimate', 'pred_LSTM_MM_WC', 'pred_LSTM_MM_NC', "ATTACKED", 'pred']]
            # df = df.sort_values(by=['k_estimate'], ascending=False)
            # print(df)
            # print(df.columns)
            # sys.exit()
    output_path = "./Output/Data_Stream/data_stream_final.csv"
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
