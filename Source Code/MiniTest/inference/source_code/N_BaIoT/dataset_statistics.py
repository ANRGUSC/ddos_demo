import math

import matplotlib.pyplot as plt
import pandas as pd
import random
import sys
import os

sys.path.append("../")
sys.path.append("../../../")
sys.path.append("./source_code/")

import project_config as CONFIG


def prepare_output_directory(output_path):
    """Prepare the output directory by deleting the old files and create an empty directory.

    Keyword arguments:
    output_path -- path to the output directory
    """
    dir_name = str(os.path.dirname(output_path))
    # os.system('rm -rf ' + dir_name)
    os.system("mkdir -p " + dir_name)


def generate_stat_attack():
    """Plot the attacked dataset statistics"""
    dataset_main_path = CONFIG.DATASET_DIRECTORY + "N_BaIoT/main_dataset/"
    device_names = [
        "Danmini_Doorbell",
        "Ecobee_Thermostat",
        "Ennio_Doorbell",
        "Philips_B120N10_Baby_Monitor",
        "Provision_PT_737E_Security_Camera",
        "Provision_PT_838_Security_Camera",
        "Samsung_SNH_1011_N_Webcam",
        "SimpleHome_XCS7_1002_WHT_Security_Camera",
        "SimpleHome_XCS7_1003_WHT_Security_Camera",
    ]

    devices_with_gafgyt_attacks = [
        "Danmini_Doorbell",
        "Ecobee_Thermostat",
        "Ennio_Doorbell",
        "Philips_B120N10_Baby_Monitor",
        "Provision_PT_737E_Security_Camera",
        "Provision_PT_838_Security_Camera",
        "Samsung_SNH_1011_N_Webcam",
        "SimpleHome_XCS7_1002_WHT_Security_Camera",
        "SimpleHome_XCS7_1003_WHT_Security_Camera",
    ]

    devices_with_mirai_attacks = [
        "Danmini_Doorbell",
        "Ecobee_Thermostat",
        "Philips_B120N10_Baby_Monitor",
        "Provision_PT_737E_Security_Camera",
        "Provision_PT_838_Security_Camera",
        "SimpleHome_XCS7_1002_WHT_Security_Camera",
        "SimpleHome_XCS7_1003_WHT_Security_Camera",
    ]

    gafgyt_attacks = ["combo", "junk", "scan", "tcp", "udp"]
    mirai_attacks = ["ack", "scan", "syn", "udp", "udpplain"]

    aggregation_methods = {
        "Source-IP": "H",
        "Channel": "HH",
        "Socket": "HpHp",
        "Source-MAC-IP": "MI_dir",
    }  # , 'Channel Jitter': 'HH_jit'
    time_windows = {
        "100-ms": "L5",
        "500-ms": "L3",
        "1.5-sec": "L1",
        "10-sec": "L0.1",
        "1-min": "L0.01",
    }
    statistics = {"Number": "weight"}

    for device_name in devices_with_gafgyt_attacks:
        folder_attack_path = dataset_main_path + device_name + "/" + "gafgyt_attacks/"
        total_attack_dataset = pd.DataFrame()
        for attack in gafgyt_attacks:
            dataset_attack_path = folder_attack_path + attack + ".csv"
            attacked_dataset = pd.read_csv(dataset_attack_path)

            for aggregation, aggregation_symbol in aggregation_methods.items():
                for time_window, time_window_symbol in time_windows.items():
                    for stat, stat_symbol in statistics.items():
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
                            + " - Gafgyt Attack: "
                            + attack
                            + "\n"
                            "Aggregation: "
                            + aggregation
                            + " - Time Window: "
                            + time_window
                        )
                        plt.title(title)
                        output_path = (
                            CONFIG.OUTPUT_DIRECTORY
                            + "N_BaIoT/all_stats/"
                            + device_name
                            + "/gafgyt/"
                            + "attack_"
                            + attack
                            + "_aggregation_"
                            + aggregation
                            + "_time_window_"
                            + time_window
                            + "_stat_"
                            + stat
                            + ".png"
                        )
                        prepare_output_directory(output_path)
                        plt.savefig(output_path)

    for device_name in devices_with_mirai_attacks:
        folder_attack_path = dataset_main_path + device_name + "/" + "mirai_attacks/"
        total_attack_dataset = pd.DataFrame()
        for attack in mirai_attacks:
            dataset_attack_path = folder_attack_path + attack + ".csv"
            attacked_dataset = pd.read_csv(dataset_attack_path)

            for aggregation, aggregation_symbol in aggregation_methods.items():
                for time_window, time_window_symbol in time_windows.items():
                    for stat, stat_symbol in statistics.items():
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
                            + attack
                            + "\n"
                            "Aggregation: "
                            + aggregation
                            + " - Time Window: "
                            + time_window
                        )
                        plt.title(title)
                        output_path = (
                            CONFIG.OUTPUT_DIRECTORY
                            + "N_BaIoT/all_stats/"
                            + device_name
                            + "/mirai/"
                            + "attack_"
                            + attack
                            + "_aggregation_"
                            + aggregation
                            + "_time_window_"
                            + time_window
                            + "_stat_"
                            + stat
                            + ".png"
                        )
                        prepare_output_directory(output_path)
                        plt.savefig(output_path)


def generate_stat_benign():
    """Plot the benign dataset statistics"""
    dataset_main_path = CONFIG.DATASET_DIRECTORY + "N_BaIoT/main_dataset/"
    device_names = [
        "Danmini_Doorbell",
        "Ecobee_Thermostat",
        "Ennio_Doorbell",
        "Philips_B120N10_Baby_Monitor",
        "Provision_PT_737E_Security_Camera",
        "Provision_PT_838_Security_Camera",
        "Samsung_SNH_1011_N_Webcam",
        "SimpleHome_XCS7_1002_WHT_Security_Camera",
        "SimpleHome_XCS7_1003_WHT_Security_Camera",
    ]

    devices_with_gafgyt_attacks = [
        "Danmini_Doorbell",
        "Ecobee_Thermostat",
        "Ennio_Doorbell",
        "Philips_B120N10_Baby_Monitor",
        "Provision_PT_737E_Security_Camera",
        "Provision_PT_838_Security_Camera",
        "Samsung_SNH_1011_N_Webcam",
        "SimpleHome_XCS7_1002_WHT_Security_Camera",
        "SimpleHome_XCS7_1003_WHT_Security_Camera",
    ]

    devices_with_mirai_attacks = [
        "Danmini_Doorbell",
        "Ecobee_Thermostat",
        "Philips_B120N10_Baby_Monitor",
        "Provision_PT_737E_Security_Camera",
        "Provision_PT_838_Security_Camera",
        "SimpleHome_XCS7_1002_WHT_Security_Camera",
        "SimpleHome_XCS7_1003_WHT_Security_Camera",
    ]

    gafgyt_attacks = ["combo", "junk", "scan", "tcp", "udp"]
    mirai_attacks = ["ack", "scan", "syn", "udp", "udpplain"]

    aggregation_methods = {
        "Source-IP": "H",
        "Channel": "HH",
        "Socket": "HpHp",
        "Source-MAC-IP": "MI_dir",
    }  # , 'Channel Jitter': 'HH_jit'
    time_windows = {
        "100-ms": "L5",
        "500-ms": "L3",
        "1.5-sec": "L1",
        "10-sec": "L0.1",
        "1-min": "L0.01",
    }
    statistics = {"Number": "weight"}

    for device_name in device_names:
        dataset_benign_traffic_path = (
            dataset_main_path + device_name + "/benign_traffic.csv"
        )
        attacked_dataset = pd.read_csv(dataset_benign_traffic_path)

        for aggregation, aggregation_symbol in aggregation_methods.items():
            for time_window, time_window_symbol in time_windows.items():
                for stat, stat_symbol in statistics.items():
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
                        + " - Benign Traffic"
                        + "\n"
                        + "Aggregation: "
                        + aggregation
                        + " - Time Window: "
                        + time_window
                    )
                    plt.title(title)
                    output_path = (
                        CONFIG.OUTPUT_DIRECTORY
                        + "N_BaIoT/all_stats/"
                        + device_name
                        + "/benign/benign_traffic_aggregation_"
                        + aggregation
                        + "_time_window_"
                        + time_window
                        + "_stat_"
                        + stat
                        + ".png"
                    )
                    prepare_output_directory(output_path)
                    plt.savefig(output_path)


if __name__ == "__main__":
    generate_stat_benign()
    generate_stat_attack()
