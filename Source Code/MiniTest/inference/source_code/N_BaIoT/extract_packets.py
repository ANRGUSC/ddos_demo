import math
import matplotlib.pyplot as plt
import pandas as pd
import random
import sys
import os

sys.path.append("../")
import project_config as CONFIG
import dataset_parameters as PARAM


def prepare_output_directory(output_path):
    """Prepare the output directory by deleting the old files and create an empty directory.

    Keyword arguments:
    output_path -- path to the output directory
    """
    dir_name = str(os.path.dirname(output_path))
    # os.system('rm -rf ' + dir_name)
    os.system("mkdir -p " + dir_name)


def plot_packets_stats_benign(
    dataset_main_path,
    device_name_list,
    aggregation_dict,
    time_window_dict,
    stat_dict,
    output_path,
):
    """Plot the benign dataset stats

    Keyword arguments:
    dataset_main_path -- path to the attacked dataset for generating the statistics
    device_name_list -- list of devices for generating the statistics in the N-BaIoT paper
    aggregation_dict -- a dictionary of the aggregation methods used in the N-BaIoT paper
    time_window_dict -- a dictionary of time windows used in the N-BaIoT paper for generating the statistics
    stat_dict -- a dictionary of the selected statistics metrics in the N-BaIoT paper for plotting them
    output_path -- path to store the plots
    """
    for device_name in device_name_list:
        dataset_benign_traffic_path = (
            dataset_main_path + device_name + "/benign_traffic.csv"
        )
        benign_dataset = pd.read_csv(dataset_benign_traffic_path)

        for aggregation, aggregation_symbol in aggregation_dict.items():
            for time_window, time_window_symbol in time_window_dict.items():
                for stat, stat_symbol in stat_dict.items():
                    column_name = (
                        aggregation_symbol
                        + "_"
                        + time_window_symbol
                        + "_"
                        + stat_symbol
                    )
                    plt.clf()
                    plt.hist(benign_dataset[column_name], bins=50)
                    plt.xlabel("Packet Count")
                    plt.ylabel("Number of Entries")
                    title = (
                        "Device: "
                        + device_name
                        + " - Benign Traffic"
                        + "\n"
                        + "Aggregation: "
                        + aggregation
                        + " - Time Window: "
                        + time_window
                    )
                    plt.title(title)
                    output_path_plot = (
                        output_path
                        + device_name
                        + "_benign_aggregation_"
                        + aggregation
                        + "_time_window_"
                        + time_window
                        + "_stat_"
                        + stat
                        + ".png"
                    )
                    plt.savefig(output_path_plot)


def plot_packets_stats_attack(
    dataset_main_path,
    device_name_list,
    attack_type_list,
    aggregation_dict,
    time_window_dict,
    stat_dict,
    output_path,
):
    """Plot the attacked dataset stats

    Keyword arguments:
    dataset_main_path -- path to the attacked dataset for generating the statistics
    device_name_list -- list of devices for generating the statistics in the N-BaIoT paper
    aggregation_dict -- a dictionary of the aggregation methods used in the N-BaIoT paper
    time_window_dict -- a dictionary of time windows used in the N-BaIoT paper for generating the statistics
    stat_dict -- a dictionary of the selected statistics metrics in the N-BaIoT paper for plotting them
    output_path -- path to store the plots
    """
    for device_name in device_name_list:
        folder_attack_path = dataset_main_path + device_name + "/" + "mirai_attacks/"
        for attack_type in attack_type_list:
            dataset_attack_path = folder_attack_path + attack_type + ".csv"
            attacked_dataset = pd.read_csv(dataset_attack_path)

            for aggregation, aggregation_symbol in aggregation_dict.items():
                for time_window, time_window_symbol in time_window_dict.items():
                    for stat, stat_symbol in stat_dict.items():
                        column_name = (
                            aggregation_symbol
                            + "_"
                            + time_window_symbol
                            + "_"
                            + stat_symbol
                        )
                        plt.clf()
                        plt.hist(attacked_dataset[column_name], bins=50)
                        plt.xlabel("Packet Count")
                        plt.ylabel("Number of Entries")
                        title = (
                            "Device: "
                            + device_name
                            + " - Mirai Attack: "
                            + attack_type
                            + "\n"
                            "Aggregation: "
                            + aggregation
                            + " - Time Window: "
                            + time_window
                        )
                        plt.title(title)
                        output_path_plot = (
                            output_path
                            + device_name
                            + "_mirai_attack_"
                            + attack_type
                            + "_aggregation_"
                            + aggregation
                            + "_time_window_"
                            + time_window
                            + "_stat_"
                            + stat
                            + ".png"
                        )
                        plt.savefig(output_path_plot)


def extract_packets_stats_benign(
    dataset_main_path,
    device_name_list,
    aggregation_dict,
    time_window_dict,
    stat_dict,
    output_path,
):
    """generate the benign dataset stats

    Keyword arguments:
    dataset_main_path -- path to the attacked dataset for generating the statistics
    device_name_list -- list of devices for generating the statistics in the N-BaIoT paper
    aggregation_dict -- a dictionary of the aggregation methods used in the N-BaIoT paper
    time_window_dict -- a dictionary of time windows used in the N-BaIoT paper for generating the statistics
    stat_dict -- a dictionary of the selected statistics metrics in the N-BaIoT paper
    output_path -- path to store the stats
    """
    for device_name in device_name_list:
        dataset_benign_traffic_path = (
            dataset_main_path + device_name + "/benign_traffic.csv"
        )
        benign_dataset = pd.read_csv(dataset_benign_traffic_path)

        for aggregation, aggregation_symbol in aggregation_dict.items():
            for time_window, time_window_symbol in time_window_dict.items():
                for stat, stat_symbol in stat_dict.items():
                    column_name = (
                        aggregation_symbol
                        + "_"
                        + time_window_symbol
                        + "_"
                        + stat_symbol
                    )
                    benign_dataset = benign_dataset.rename(
                        columns={column_name: "PACKET"}
                    )
                    benign_dataset = benign_dataset[["PACKET"]]

                    output_path_data = (
                        output_path
                        + device_name
                        + "_benign_aggregation_"
                        + aggregation
                        + "_time_window_"
                        + time_window
                        + "_stat_"
                        + stat
                        + ".csv"
                    )
                    benign_dataset.to_csv(output_path_data, index=False)


def extract_packets_stats_attack(
    dataset_main_path,
    device_name_list,
    attack_type_list,
    aggregation_dict,
    time_window_dict,
    stat_dict,
    output_path,
):
    """generate the benign dataset stats

    Keyword arguments:
    dataset_main_path -- path to the attacked dataset for generating the statistics
    device_name_list -- list of devices for generating the statistics in the N-BaIoT paper
    aggregation_dict -- a dictionary of the aggregation methods used in the N-BaIoT paper
    time_window_dict -- a dictionary of time windows used in the N-BaIoT paper for generating the statistics
    stat_dict -- a dictionary of the selected statistics metrics in the N-BaIoT paper
    output_path -- path to store the stats
    """
    for device_name in device_name_list:
        folder_attack_path = dataset_main_path + device_name + "/" + "mirai_attacks/"
        for attack_type in attack_type_list:
            dataset_attack_path = folder_attack_path + attack_type + ".csv"
            attacked_dataset = pd.read_csv(dataset_attack_path)

            for aggregation, aggregation_symbol in aggregation_dict.items():
                for time_window, time_window_symbol in time_window_dict.items():
                    for stat, stat_symbol in stat_dict.items():
                        column_name = (
                            aggregation_symbol
                            + "_"
                            + time_window_symbol
                            + "_"
                            + stat_symbol
                        )
                        attacked_dataset = attacked_dataset.rename(
                            columns={column_name: "PACKET"}
                        )
                        attacked_dataset = attacked_dataset[["PACKET"]]

                        output_path_data = (
                            output_path
                            + device_name
                            + "_mirai_attack_"
                            + attack_type
                            + "_aggregation_"
                            + aggregation
                            + "_time_window_"
                            + time_window
                            + "_stat_"
                            + stat
                            + ".csv"
                        )
                        attacked_dataset.to_csv(output_path_data, index=False)


def main_packets_stats():
    """main function for generating and plotting the benign/attacked dataset"""

    dataset_main_path = CONFIG.DATASET_DIRECTORY + "N_BaIoT/main_dataset/"

    device_name_list = ["SimpleHome_XCS7_1003_WHT_Security_Camera"]
    # device_name_list = PARAM.devices_with_mirai_attacks
    attack_type_list = ["udp"]
    aggregation_dict = {"Source-IP": "H"}
    time_window_dict = {"10-sec": "L0.1"}
    stat_dict = {"Number": "weight"}

    output_path = CONFIG.OUTPUT_DIRECTORY + "N_BaIoT/Plot/"
    prepare_output_directory(output_path)
    plot_packets_stats_attack(
        dataset_main_path,
        device_name_list,
        attack_type_list,
        aggregation_dict,
        time_window_dict,
        stat_dict,
        output_path,
    )

    plot_packets_stats_benign(
        dataset_main_path,
        device_name_list,
        aggregation_dict,
        time_window_dict,
        stat_dict,
        output_path,
    )

    output_path = CONFIG.OUTPUT_DIRECTORY + "N_BaIoT/Data/"
    prepare_output_directory(output_path)
    extract_packets_stats_attack(
        dataset_main_path,
        device_name_list,
        attack_type_list,
        aggregation_dict,
        time_window_dict,
        stat_dict,
        output_path,
    )

    extract_packets_stats_benign(
        dataset_main_path,
        device_name_list,
        aggregation_dict,
        time_window_dict,
        stat_dict,
        output_path,
    )


if __name__ == "__main__":
    main_packets_stats()
