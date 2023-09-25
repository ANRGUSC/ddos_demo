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
    os.system("rm -rf " + dir_name)
    os.system("mkdir -p " + dir_name)


def load_dataset(path):
    """Load the dataset

    Keyword arguments:
    path -- path to the dataset csv file
    """
    data = pd.read_csv(path)
    data["TIME"] = pd.to_datetime(data["TIME"])
    return data


def ensemble_prediction(
    dense_prediction_path,
    cnn_prediction_path,
    lstm_prediction_path,
    aen_prediction_path,
    data_type,
    output_path,
):
    dense = pd.read_csv(dense_prediction_path)
    cnn = pd.read_csv(cnn_prediction_path)
    lstm = pd.read_csv(lstm_prediction_path)
    aen = pd.read_csv(aen_prediction_path)

    dense = dense[
        ["ATTACK_PARAMETER", "NODE", "TIME", "TRUE", "PRED", "TP", "FP"]
    ].sort_values(by=["ATTACK_PARAMETER", "NODE", "TIME"])
    cnn = cnn[
        ["ATTACK_PARAMETER", "NODE", "TIME", "TRUE", "PRED", "TP", "FP"]
    ].sort_values(by=["ATTACK_PARAMETER", "NODE", "TIME"])
    lstm = lstm[
        ["ATTACK_PARAMETER", "NODE", "TIME", "TRUE", "PRED", "TP", "FP"]
    ].sort_values(by=["ATTACK_PARAMETER", "NODE", "TIME"])
    aen = aen[
        ["ATTACK_PARAMETER", "NODE", "TIME", "TRUE", "PRED", "TP", "FP"]
    ].sort_values(by=["ATTACK_PARAMETER", "NODE", "TIME"])

    models_df = dense.copy()
    models_df = models_df.rename(
        columns={"PRED": "DENSE_PRED", "TP": "DENSE_TP", "FP": "DENSE_FP"}
    )
    models_df["TRUE"] = models_df["TRUE"].astype(int)
    models_df["DENSE_INCORRECT_PRED"] = models_df["DENSE_PRED"] ^ models_df["TRUE"]
    models_df["CNN_PRED"] = cnn["PRED"]
    models_df["CNN_TP"] = cnn["TP"]
    models_df["CNN_FP"] = cnn["FP"]
    models_df["CNN_INCORRECT_PRED"] = models_df["CNN_PRED"] ^ models_df["TRUE"]
    models_df["LSTM_PRED"] = lstm["PRED"]
    models_df["LSTM_TP"] = lstm["TP"]
    models_df["LSTM_FP"] = lstm["FP"]
    models_df["LSTM_INCORRECT_PRED"] = models_df["LSTM_PRED"] ^ models_df["TRUE"]
    models_df["AEN_PRED"] = aen["PRED"]
    models_df["AEN_TP"] = aen["TP"]
    models_df["AEN_FP"] = aen["FP"]
    models_df["AEN_INCORRECT_PRED"] = models_df["AEN_PRED"] ^ models_df["TRUE"]

    models_df["ENS_PRED"] = (
        models_df["DENSE_PRED"]
        + models_df["CNN_PRED"]
        + models_df["LSTM_PRED"]
        + models_df["AEN_PRED"]
    )
    models_df["ENS_PRED"] = (models_df["ENS_PRED"] >= 2).astype(int)
    models_df["ENS_INCORRECT_PRED"] = models_df["ENS_PRED"] ^ models_df["TRUE"]
    ens_acc = 1 - models_df["ENS_INCORRECT_PRED"].mean()
    models_df["ENS_TP"] = models_df["TRUE"] & models_df["ENS_PRED"]
    ens_tp = models_df["ENS_TP"].sum() / models_df["TRUE"].sum()
    print(ens_acc)
    print(ens_tp)

    k_list = models_df["ATTACK_PARAMETER"].unique()
    result_df = pd.DataFrame()
    for k in k_list:
        temp = models_df.loc[models_df["ATTACK_PARAMETER"] == k]

        dense_acc = 1 - temp["DENSE_INCORRECT_PRED"].mean()
        dense_recall = temp["DENSE_TP"].sum() / temp["TRUE"].sum()
        cnn_acc = 1 - temp["CNN_INCORRECT_PRED"].mean()
        cnn_recall = temp["CNN_TP"].sum() / temp["TRUE"].sum()
        lstm_acc = 1 - temp["LSTM_INCORRECT_PRED"].mean()
        lstm_recall = temp["LSTM_TP"].sum() / temp["TRUE"].sum()
        aen_acc = 1 - temp["AEN_INCORRECT_PRED"].mean()
        aen_recall = temp["AEN_TP"].sum() / temp["TRUE"].sum()
        ens_acc = 1 - temp["ENS_INCORRECT_PRED"].mean()
        ens_recall = temp["ENS_TP"].sum() / temp["TRUE"].sum()

        row = {
            "ATTACK_PARAMETER": k,
            "DATA_TYPE": data_type,
            "MLP_ACC": dense_acc,
            "MLP_RECALL": dense_recall,
            "CNN_ACC": cnn_acc,
            "CNN_RECALL": cnn_recall,
            "LSTM_ACC": lstm_acc,
            "LSTM_RECALL": lstm_recall,
            "AEN_ACC": aen_acc,
            "AEN_RECALL": aen_recall,
            "ENS_ACC": ens_acc,
            "ENS_RECALL": ens_recall,
        }

        result_df = result_df.append(row, ignore_index=True)

    result_df.to_csv(output_path, index=False)
    return result_df


def plot_compare_ensemble_model(data, output_path):
    # k_list = data['ATTACK_PARAMETER'].unique()
    colors = {0: "black", 1: "red", 2: "blue", 3: "green", 4: "yellow", 5: "magenta"}
    line_styles = {"train": "solid", "test": "dashed"}
    markers = {0: ".", 1: "v", 2: "+", 3: "x", 4: "s", 5: "^"}
    model_legends = {0: "MLP", 1: "CNN", 2: "LSTM", 3: "AEN", 4: "ENS"}

    data_type = data["DATA_TYPE"][0]

    plt.clf()
    for i in range(5):
        metric = model_legends[i] + "_ACC"
        plt.plot(
            data["ATTACK_PARAMETER"],
            data[metric],
            color=colors[i],
            linestyle=line_styles[data_type],
            marker=markers[i],
            label=model_legends[i],
        )
    plt.legend()
    plt.ylim([0.7, 1])
    plt.xlabel("Attack Packet Volume Distribution Parameter (k)")
    plt.ylabel("Binary Accuracy")
    output_path_acc = output_path + "models_comparison_acc_" + data_type + "_data.png"
    plt.savefig(output_path_acc, bbox_inches="tight")

    plt.clf()
    for i in range(5):
        metric = model_legends[i] + "_RECALL"
        plt.plot(
            data["ATTACK_PARAMETER"],
            data[metric],
            color=colors[i],
            linestyle=line_styles[data_type],
            marker=markers[i],
            label=model_legends[i],
        )
    plt.legend()
    plt.ylim([0.2, 1.05])
    plt.xlabel("Attack Packet Volume Distribution Parameter (k)")
    plt.ylabel("Recall")
    output_path_recall = (
        output_path + "models_comparison_recall_" + data_type + "_data.png"
    )
    plt.savefig(output_path_recall, bbox_inches="tight")


def main():
    seed = 1
    tf.random.set_seed(seed)
    random.seed(seed)

    metric = "val_binary_accuracy"
    mode = "max"
    k_list = np.array([0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1])
    k_list = np.round(k_list, 2)
    time_window = 10

    upsample = "with"
    model_architecture = "one_model_no_correlation_one_hot"

    train_file_name = "train_result_" + metric + "_" + mode + ".csv"
    test_file_name = "test_result_" + metric + "_" + mode + ".csv"

    dense_prediction_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training_dense/current_features_aggregate_all_k/Output_"
        + upsample
        + "_upsample_test/"
        + model_architecture
        + "/attack_prediction_vs_time/data/"
    )

    cnn_prediction_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training_cnn/current_features_aggregate_all_k/Output_"
        + upsample
        + "_upsample_test/"
        + model_architecture
        + "/attack_prediction_vs_time/data/"
    )

    lstm_prediction_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training_lstm/current_features_aggregate_all_k/Output_"
        + upsample
        + "_upsample_test/"
        + model_architecture
        + "/attack_prediction_vs_time/data/"
    )

    aen_prediction_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "nn_training_autoencoder/current_features_aggregate_all_k/Output_"
        + upsample
        + "_upsample_test/"
        + model_architecture
        + "/attack_prediction_vs_time/data/"
    )

    dense_prediction_train_path = dense_prediction_path + train_file_name
    cnn_prediction_train_path = cnn_prediction_path + train_file_name
    lstm_prediction_train_path = lstm_prediction_path + train_file_name
    aen_prediction_train_path = aen_prediction_path + train_file_name

    dense_prediction_test_path = dense_prediction_path + test_file_name
    cnn_prediction_test_path = cnn_prediction_path + test_file_name
    lstm_prediction_test_path = lstm_prediction_path + test_file_name
    aen_prediction_test_path = aen_prediction_path + test_file_name

    plot_output_path = "./Output/plot/"
    prepare_output_directory(plot_output_path)

    models_result_training_data_path = "./Output/data/models_result_training_data.csv"
    prepare_output_directory(models_result_training_data_path)
    models_training_data = ensemble_prediction(
        dense_prediction_train_path,
        cnn_prediction_train_path,
        lstm_prediction_train_path,
        aen_prediction_train_path,
        "train",
        models_result_training_data_path,
    )

    plot_compare_ensemble_model(models_training_data, plot_output_path)

    models_result_testing_data_path = "./Output/data/models_result_testing_data.csv"
    models_testing_data = ensemble_prediction(
        dense_prediction_test_path,
        cnn_prediction_test_path,
        lstm_prediction_test_path,
        aen_prediction_test_path,
        "test",
        models_result_testing_data_path,
    )

    plot_compare_ensemble_model(models_testing_data, plot_output_path)


if __name__ == "__main__":
    main()
