#!/usr/bin/env python3
import argparse
import threading
import time

import benign_behavior
import node_udp_server
import node_udp_client
import ddos_attack_simulation
import network_flow_capture
from datetime import datetime
from data_processing import data_processing
import socket
import pickle
import json
import subprocess


def ddos_experiment(traffic_file_name):
    set_up_dic, server_ip = active_start_server()
    print('Start Tracking')
    treads_dic = dict()

    json_file_path = './shared_variable/shared_variable.json'
    json_variable = {
        "ACTIVE": False,
        "ATTACKED": False
    }
    with open(json_file_path, 'w') as file:
        json.dump(json_variable, file)
    t = threading.Thread(target=node_udp_server.udp_server, args=(node_udp_server.get_ip_address(),
                                                                  9564,
                                                                  set_up_dic['active_time'],
                                                                  datetime.now(),
                                                                  set_up_dic['node_number'],
                                                                  set_up_dic['attack_ratio'],
                                                                  datetime.now() + set_up_dic['duration'],
                                                                  # attack start
                                                                  set_up_dic['duration'],
                                                                  set_up_dic['k'],
                                                                  set_up_dic['ip_list'],
                                                                  set_up_dic['victim_ip']
                                                                  ))
    t.daemon = True
    t.start()
    treads_dic['udp_server'] = t

    client = threading.Thread(target=node_udp_client.client, args=(set_up_dic['active_time'], set_up_dic['ip_list']))
    client.daemon = True
    client.start()
    treads_dic['udp_client'] = client

    packet_capture = threading.Thread(target=network_flow_capture.network_capture, args=(traffic_file_name,))
    packet_capture.daemon = True
    packet_capture.start()
    treads_dic['packet_capture'] = packet_capture

    print('Start Experiment')
    ret_dic = node_time_control(json_file_path)
    print('Join Threads')
    for key in treads_dic:
        print('Join Thread ', key)
        treads_dic[key].join(timeout=2)
        print('Thread ', key, ' ends')

    # node_ip, start_time, node_num, attack_ratio, attack_start, attack_end, duration, k, active_list,
    # ddos_list
    print('Start Processing Data')
    current_time = datetime.now()
    current_time = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')
    formatted_time = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S.%f')
    subprocess.run(['sudo rm -f /home/jiahe/MiniTest/inference/{}_prediction_data.csv'.format(set_up_dic['node_ip'][-3:])], check=True, shell=True)
    data_processing(set_up_dic['node_ip'],
                    ret_dic['start_time'],
                    set_up_dic['node_number'],
                    set_up_dic['attack_ratio'],
                    ret_dic['attack_start'],  # attack start
                    formatted_time,
                    set_up_dic['duration'],
                    set_up_dic['k'],
                    ret_dic['active_list'],
                    ret_dic['attack_time'],
                    set_up_dic['active_time'],
                    set_up_dic['victim_ip'],
                    set_up_dic['ip_list']
                    )
    send_end_signal(server_ip)
    print('Experiment Ends')


def node_time_control(json_file_path):
    start_time = datetime.now()

    active_time_list = []
    ddos_time_list = []
    s_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s_socket.bind((node_udp_server.get_ip_address(), 9565))
    while True:
        data, addr = s_socket.recvfrom(2048)
        received_array = pickle.loads(data)
        print(received_array['clk'])
        if received_array['is_end']:
            print('Experiments End')
            attack_start = received_array['attack_start']
            break
        if received_array['is_ddos']:
            print("DDOS start")
            current_time = datetime.now()
            formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')
            ddos_time_list.append(datetime.strptime(formatted_time, "%Y-%m-%d %H:%M:%S.%f"))
            variable_operation(json_file_path, {"ATTACKED": True})
            ddos_attack_simulation.ddos_attack(received_array['victim_ip'],
                                               received_array['active_time'],
                                               received_array['k'],
                                               received_array['sleep'])
            variable_operation(json_file_path, {"ATTACKED": False})
        else:
            print('Benign behavior')
            current_time = datetime.now()
            formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')
            active_time_list.append(datetime.strptime(formatted_time, "%Y-%m-%d %H:%M:%S.%f"))
            variable_operation(json_file_path, {"ACTIVE": True})
            benign_behavior.udp_send(received_array['active_time'], received_array['sleep'])
            variable_operation(json_file_path, {"ACTIVE": False})
        print('---------------')
    return {'attack_time': ddos_time_list, 'active_list': active_time_list, 'start_time': start_time,
            'attack_start': attack_start}

    # active_time, start_min, end_min, prec, k, sleep


def variable_operation(file_path, chg_data):
    with open(file_path, 'r') as file:
        data = json.load(file)
    for key in chg_data:
        data[key] = chg_data[key]
    with open(file_path, 'w') as file:
        json.dump(data, file)


def active_start_server():
    s_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s_socket.bind((node_udp_server.get_ip_address(), 9565))
    while True:
        data, addr = s_socket.recvfrom(2048)
        received_array = pickle.loads(data)
        print(received_array)
        if received_array is not None:
            return received_array, addr[0]


def send_end_signal(server_ip):
    print('sending signal to ', server_ip)
    for i in range(10):
        c_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        c_socket.sendto(pickle.dumps([True]), (server_ip, 9568))
        time.sleep(0.5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='traffic')
    args = parser.parse_args()
    ddos_experiment(args.name)
