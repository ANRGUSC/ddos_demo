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
from datetime import timedelta

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
    data = pd.read_csv(path)
    data["TIME"] = pd.to_datetime(data["TIME"])
    return data


def plot_attack_prediction_vs_time(data, output_path):
    attack_ratios = list(data["ATTACK_RATIO"].unique())
    attack_durations = list(data["ATTACK_DURATION"].unique())
    print(attack_ratios)
    print(attack_durations)

    k = 1
    attack_ratio = 1
    attack_duration = "0 days 16:00:00"

    selected_data = data.loc[
        (data["ATTACK_RATIO"] == attack_ratio)
        & (data["ATTACK_DURATION"] == attack_duration)
        & (data["ATTACK_PARAMETER"] == k)
    ]

    selected_data = selected_data.groupby(["TIME"]).mean().reset_index()
    selected_data = selected_data.sort_values(by=["TIME"])

    time_window = timedelta(hours=4)
    time_step = timedelta(hours=1)
    begin_time = selected_data["TIME"][0]
    end_time = selected_data["TIME"][selected_data.shape[0] - 1]
    cnt = 0
    while begin_time < end_time - time_window:
        plot_data = selected_data.loc[
            (selected_data["TIME"] >= begin_time)
            & (selected_data["TIME"] < begin_time + time_window)
        ]
        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(plot_data["TIME"], plot_data["TRUE"], label="True")
        ax.plot(plot_data["TIME"], plot_data["TP"], label="TP")
        ax.plot(plot_data["TIME"], plot_data["FP"], label="FP")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        myFmt = mdates.DateFormatter("%H")
        ax.xaxis.set_major_formatter(myFmt)
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Attack")
        ax.set_ylim([0, 1.05])

        output_path_temp = (
            output_path
            + "test_attack_prediction_vs_time_"
            + str(attack_duration)
            + "_attackRatio_"
            + str(attack_ratio)
            + "_duration_"
            + str(attack_duration)
            + "_k_"
            + str(k)
            + "_"
            + str(cnt)
            + ".png"
        )
        fig.savefig(output_path_temp)
        cnt += 1
        begin_time += time_step


def main_plot_attack_prediction_vs_time():
    dataset_path = "./dataset/LSTM_Test_Prediction_Vs_Time.csv"
    output_path = "./Output/Attack_Prediction_VS_Time/"
    prepare_output_directory(output_path)

    data = load_dataset(dataset_path)

    plot_attack_prediction_vs_time(data, output_path)


def main():
    seed = 1
    tf.random.set_seed(seed)
    random.seed(seed)

    metric = "val_binary_accuracy"
    mode = "max"
    k_list = np.array([0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1])

    k_list = np.round(k_list, 2)
    time_window = 10

    main_plot_attack_prediction_vs_time()


if __name__ == "__main__":
    main()
