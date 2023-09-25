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
    accuracy_score,
    recall_score,
)
import random
import os
from multiprocessing import Pool, Manager
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pickle import load
from scipy.stats import norm, cauchy
from datetime import datetime, timedelta

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


def data_stream(ns, node, k, attack_ratio, attack_duration, results_df, df):
    print(node, " , ", k, " , ", attack_ratio, " , ", attack_duration)

    data = results_df.loc[
        (results_df["NODE"] == node)
        & (results_df["ATTACK_PARAMETER"] == k)
        & (results_df["ATTACK_RATIO"] == attack_ratio)
        & (results_df["ATTACK_DURATION"] == attack_duration)
    ]
    if data.shape[0] == 0:
        return
    data = data.sort_values(by=["TIME"]).reset_index(drop=True)

    benign_que = collections.deque(maxlen=10000)
    attack_que = collections.deque(maxlen=10000)
    for index, row in data.iterrows():
        if data.loc[index, "PACKET"] > 0 and data.loc[index, "init_pred"] == 0:
            benign_que.append(data.loc[index, "PACKET"])
        elif data.loc[index, "PACKET"] > 0 and data.loc[index, "init_pred"] == 1:
            attack_que.append(data.loc[index, "PACKET"])

        if len(benign_que) > 0 and len(attack_que) > 0:
            benign_parameter = cauchy.fit(list(benign_que))
            attack_parameter = cauchy.fit(list(attack_que))

            k_estimate_1 = max(attack_parameter[0] / benign_parameter[0] - 1, 0)
            k_estimate_2 = max(attack_parameter[1] / benign_parameter[1] - 1, 0)
            k_estimate = (k_estimate_1 + k_estimate_2) / 2
            if k_estimate < 0.3:
                pred = data.loc[index, "pred_LSTM_MM_WC"]
            else:
                pred = data.loc[index, "pred_LSTM_MM_NC"]
        else:
            k_estimate = 0
            pred = data.loc[index, "pred_LSTM_MM_WC"]

        data.loc[index, "k_estimate"] = k_estimate
        data.loc[index, "pred"] = pred

    df = df.append(data, ignore_index=True)
    ns.append(df)


def main_data_stream():
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
        CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/test_data/test_data_3_days.csv"
    )
    test_dataset = load_dataset(test_dataset_path)

    output_path = "./Output/Data_Stream/data_stream_"
    results_df = pd.DataFrame()
    attack_ratios = test_dataset["ATTACK_RATIO"].unique()
    attack_durations = test_dataset["ATTACK_DURATION"].unique()
    nodes = test_dataset["NODE"].unique()
    k_list = test_dataset["ATTACK_PARAMETER"].unique()
    """
    for node in nodes:
        print("****************************************************")
        print("********************  Node = ", node, "***********************")
        print(datetime.now())
        print("****************************************************")

        LSTM_MM_WC_model, LSTM_MM_WC_scaler = MM_load_model_weights(LSTM_MM_WC_model_path, node, metric, mode)
        LSTM_MM_NC_model, LSTM_MM_NC_scaler = MM_load_model_weights(LSTM_MM_NC_model_path, node, metric, mode)

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

                    X_LSTM_MM_WC, y_LSTM_MM_WC, df_LSTM_MM_WC = LSTM_MM_WC_get_input_target(data_4, num_labels, time_window, LSTM_MM_WC_scaler)
                    X_LSTM_MM_NC, y_LSTM_MM_NC, df_LSTM_MM_NC = LSTM_MM_NC_get_input_target(data_4, num_labels, time_window, LSTM_MM_NC_scaler)

                    data_4 = df_LSTM_MM_WC
                    #init_pred = np.rint(LSTM_MM_WC_model.predict(X_LSTM_MM_WC))
                    pred_LSTM_MM_WC = np.rint(LSTM_MM_WC_model.predict(X_LSTM_MM_WC))
                    pred_LSTM_MM_WC = np.reshape(pred_LSTM_MM_WC, pred_LSTM_MM_WC.shape[0])
                    pred_LSTM_MM_NC = np.rint(LSTM_MM_NC_model.predict(X_LSTM_MM_NC))
                    pred_LSTM_MM_NC = np.reshape(pred_LSTM_MM_NC, pred_LSTM_MM_NC.shape[0])
                    init_pred = pred_LSTM_MM_WC

                    data_4['pred_LSTM_MM_WC'] = pred_LSTM_MM_WC
                    data_4['pred_LSTM_MM_NC'] = pred_LSTM_MM_NC
                    data_4['init_pred'] = init_pred

                    results_df = results_df.append(data_4)
    output_path = './Output/Data_Stream/Test_Results/data_stream_final_3_days_10000_que.csv'
    prepare_output_directory(output_path)
    results_df.to_csv(output_path, index=False)
    """
    results_df = pd.read_csv(
        "./Output/Data_Stream/Test_Results/data_stream_final_3_days_1000_que.csv"
    )
    print("first done")

    columns = list(results_df.columns)
    columns.extend(["k_estimate", "pred"])
    df = pd.DataFrame(columns=columns)
    manager = Manager()
    ns = manager.list([df])

    p = Pool(6)
    p.starmap(
        data_stream,
        product(
            [ns], nodes, k_list, attack_ratios, attack_durations, [results_df], [df]
        ),
    )
    p.close()
    p.join()

    df = pd.concat(ns, ignore_index=True)

    output_path = "./Output/Data_Stream/Test_Results/data_stream_with_pred_final_3_days_10000_que.csv"
    df.to_csv(output_path, index=False)


def generate_metrics_vs_k(data, output_path):
    df = pd.DataFrame()
    k_list = data["ATTACK_PARAMETER"].unique()
    for k in k_list:
        print("k: ", k)
        temp = data.loc[data["ATTACK_PARAMETER"] == k]

        ensemble_accuracy = accuracy_score(temp["ATTACKED"], temp["pred"])
        LSTM_MM_WC_accuracy = accuracy_score(temp["ATTACKED"], temp["pred_LSTM_MM_WC"])
        LSTM_MM_NC_accuracy = accuracy_score(temp["ATTACKED"], temp["pred_LSTM_MM_NC"])

        ensemble_recall = recall_score(temp["ATTACKED"], temp["pred"])
        LSTM_MM_WC_recall = recall_score(temp["ATTACKED"], temp["pred_LSTM_MM_WC"])
        LSTM_MM_NC_recall = recall_score(temp["ATTACKED"], temp["pred_LSTM_MM_NC"])

        row = {
            "k": k,
            "ensemble_accuracy": ensemble_accuracy,
            "LSTM_MM_WC_accuracy": LSTM_MM_WC_accuracy,
            "LSTM_MM_NC_accuracy": LSTM_MM_NC_accuracy,
            "ensemble_recall": ensemble_recall,
            "LSTM_MM_WC_recall": LSTM_MM_WC_recall,
            "LSTM_MM_NC_recall": LSTM_MM_NC_recall,
        }

        df = df.append(row, ignore_index=True)
    df = df.sort_values(by=["k"])
    df.to_csv(output_path, index=False)


def plot_metrics_vs_k(data, output_path):
    plt.clf()
    plt.plot(data["k"], data["ensemble_accuracy"], label="Ensemble")
    plt.plot(data["k"], data["LSTM_MM_WC_accuracy"], label="LSTM_MM_WC")
    plt.plot(data["k"], data["LSTM_MM_NC_accuracy"], label="LSTM_MM_NC")
    plt.xlabel("Attack Parameter")
    plt.ylabel("Binary Accuracy")
    plt.title("Binary Accuracy vs attack parameter")
    plt.legend()
    plt.savefig(output_path + "accuracy_vs_k.png")

    plt.clf()
    plt.plot(data["k"], data["ensemble_recall"], label="Ensemble")
    plt.plot(data["k"], data["LSTM_MM_WC_recall"], label="LSTM_MM_WC")
    plt.plot(data["k"], data["LSTM_MM_NC_recall"], label="LSTM_MM_NC")
    plt.xlabel("Attack Parameter")
    plt.ylabel("Recall")
    plt.title("Recall vs attack parameter")
    plt.legend()
    plt.savefig(output_path + "recall_vs_k.png")


def plot_metrics_vs_time(data, output_path):
    plt.clf()
    plt.plot(data["ensemble_accuracy"], label="Ensemble")
    plt.plot(data["LSTM_MM_WC_accuracy"], label="LSTM_MM_WC")
    plt.plot(data["LSTM_MM_NC_accuracy"], label="LSTM_MM_NC")
    plt.xlabel("Time")
    plt.ylabel("Binary Accuracy")
    plt.title("Binary Accuracy vs attack parameter")
    plt.legend()
    plt.savefig(output_path + "accuracy_vs_time.png")

    plt.clf()
    plt.plot(data["ensemble_recall"], label="Ensemble")
    plt.plot(data["LSTM_MM_WC_recall"], label="LSTM_MM_WC")
    plt.plot(data["LSTM_MM_NC_recall"], label="LSTM_MM_NC")
    plt.xlabel("Time")
    plt.ylabel("Recall")
    plt.title("Recall vs attack parameter")
    plt.legend()
    plt.savefig(output_path + "recall_vs_time.png")


def generate_metrics_vs_time(data, time_step, period, output_path):
    data = data.sort_values(by=["TIME"]).reset_index(drop=True)
    begin_time = data["TIME"][0]
    end_time = data["TIME"][data.shape[0] - 1]
    print(begin_time)
    print(end_time)
    time = begin_time
    df = pd.DataFrame()
    while time + period < end_time:
        print(time + period)
        temp = data.loc[(data["TIME"] >= time) & (data["TIME"] < time + period)]

        ensemble_accuracy = accuracy_score(temp["ATTACKED"], temp["pred"])
        LSTM_MM_WC_accuracy = accuracy_score(temp["ATTACKED"], temp["pred_LSTM_MM_WC"])
        LSTM_MM_NC_accuracy = accuracy_score(temp["ATTACKED"], temp["pred_LSTM_MM_NC"])

        ensemble_recall = recall_score(temp["ATTACKED"], temp["pred"], zero_division=0)
        LSTM_MM_WC_recall = recall_score(
            temp["ATTACKED"], temp["pred_LSTM_MM_WC"], zero_division=0
        )
        LSTM_MM_NC_recall = recall_score(
            temp["ATTACKED"], temp["pred_LSTM_MM_NC"], zero_division=0
        )

        row = {
            "TIME": time + period,
            "ensemble_accuracy": ensemble_accuracy,
            "LSTM_MM_WC_accuracy": LSTM_MM_WC_accuracy,
            "LSTM_MM_NC_accuracy": LSTM_MM_NC_accuracy,
            "ensemble_recall": ensemble_recall,
            "LSTM_MM_WC_recall": LSTM_MM_WC_recall,
            "LSTM_MM_NC_recall": LSTM_MM_NC_recall,
        }
        df = df.append(row, ignore_index=True)
        time += time_step

    df = df.sort_values(by=["TIME"])
    df.to_csv(output_path, index=False)


def main_generate_metrics():
    dataset_path = "./Output/Data_Stream/Test_Results/data_stream_with_pred_final_3_days_10000_que.csv"
    data = load_dataset(dataset_path)
    data["ATTACKED"] = data["ATTACKED"].astype(int)
    data["pred"] = data["pred"].astype(int)
    data["pred_LSTM_MM_WC"] = data["pred_LSTM_MM_WC"].astype(int)
    data["pred_LSTM_MM_NC"] = data["pred_LSTM_MM_NC"].astype(int)

    output_path = "./Output/Data_Stream/Metrics/accuracy_recall_vs_k.csv"
    prepare_output_directory(output_path)
    generate_metrics_vs_k(data, output_path)

    period = timedelta(hours=1)
    time_step = timedelta(seconds=30)
    output_path = "./Output/Data_Stream/Metrics/accuracy_recall_vs_time.csv"
    generate_metrics_vs_time(data, time_step, period, output_path)


def main_plot_metrics():
    accuracy_recall_vs_k_data_path = (
        "./Output/Data_Stream/Metrics/accuracy_recall_vs_k.csv"
    )
    accuracy_recall_vs_k_dataset = pd.read_csv(accuracy_recall_vs_k_data_path)
    plot_path = "./Output/Data_Stream/Metrics/"
    plot_metrics_vs_k(accuracy_recall_vs_k_dataset, plot_path)

    accuracy_recall_vs_time_data_path = (
        "./Output/Data_Stream/Metrics/accuracy_recall_vs_time.csv"
    )
    accuracy_recall_vs_time_dataset = pd.read_csv(accuracy_recall_vs_time_data_path)
    plot_path = "./Output/Data_Stream/Metrics/"
    plot_metrics_vs_time(accuracy_recall_vs_time_dataset, plot_path)


if __name__ == "__main__":
    main_data_stream()
    main_generate_metrics()
    main_plot_metrics()
