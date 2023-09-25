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


def plot_nodes_activity(data, output_path):
    begin_time = data["TIME"][0]
    end_time = begin_time + timedelta(days=1)
    time_step = timedelta(hours=1)

    tmp = data.loc[data["TIME"] == begin_time]
    plt.clf()
    plt.scatter(tmp["LAT"], tmp["LNG"], s=1)
    plt.show()
    return
    cnt = 0
    while begin_time < end_time:
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        tmp = data.loc[data["TIME"] == begin_time]
        ax.scatter(tmp["LAT"], tmp["LNG"], tmp["ACTIVE"])
        plt.title(str(begin_time))

        plt.savefig(output_path + str(cnt) + ".png")
        # plt.show()
        # return
        begin_time += time_step
        cnt += 1

    # plt.clf()
    # plt.scatter(data['LAT_2'], data['LNG_2'], s = 1)
    # plt.show()


def plot_nodes_activity_2(data, output_path):
    begin_time = data["TIME"][0]
    end_time = begin_time + timedelta(days=2)
    time_step = timedelta(hours=1)
    data = data.loc[data["ACTIVE"] == 1]
    cnt = 0
    while begin_time < end_time:
        tmp = data.loc[(data["TIME"] == begin_time)]
        plt.clf()
        plt.scatter(tmp["LNG"], tmp["LAT"], s=4)
        plt.title("Time: " + str(begin_time))
        plt.xlim([1, 6])
        plt.ylim([4, 15])
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        plt.savefig(output_path + str(cnt) + ".png")
        # plt.show()
        # return
        begin_time += time_step
        cnt += 1

    # plt.clf()
    # plt.scatter(data['LAT_2'], data['LNG_2'], s = 1)
    # plt.show()


def main_plot_nodes_activity():
    dataset_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "/pre_process/Output/benign_data/benign_data_2021-01-02 00:00:00_2021-02-01 23:59:58_time_step_30_num_ids_4060.csv"
    )
    nodes_info_anonym_path = (
        CONFIG.OUTPUT_DIRECTORY
        + "/pre_process/Output/dataset_info/dataset_info_anonym.csv"
    )
    output_path = "./Output/Nodes_Activity/nodes_activity_"
    prepare_output_directory(output_path)

    data = load_dataset(dataset_path)
    # temp_output_path = CONFIG.OUTPUT_DIRECTORY + '/pre_process/Output/benign_data/benign_data_2021-01-02 00:00:00_2021-02-01 23:59:58_time_step_30_num_ids_4060.csv'
    # begin_time = data['TIME'][0]
    # end_time = begin_time + timedelta(days=2)
    # data_selected = data.loc[ (data['TIME'] >= begin_time) &
    #                          (data['TIME'] < end_time)]
    # data_selected.to_csv(temp_output_path, index = False)
    # sys.exit()

    nodes_info = pd.read_csv(nodes_info_anonym_path)

    plot_nodes_activity_2(data, output_path)


def main():
    main_plot_nodes_activity()


if __name__ == "__main__":
    main()
