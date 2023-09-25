#!/usr/bin/env python3
import argparse

import random
import time
import pandas as pd
import socket
import pickle
from datetime import datetime, timedelta
import paramiko
import os
import csv


def ddos_experiment(packet_volume_interval, traffic_file_name, k, attack_ratio, duration, memo):
    ip_list = [
        '192.168.1.160',
        '192.168.1.159',
        '192.168.1.166',
        '192.168.1.157',
        '192.168.1.187',
        '192.168.1.198',
        '192.168.1.108',
        '192.168.1.109',
        '192.168.1.110',
        '192.168.1.111',
        '192.168.1.112',
        '192.168.1.113',
        '192.168.1.114',
        '192.168.1.115',
        '192.168.1.116',
        '192.168.1.117',
        '192.168.1.118',
        '192.168.1.119',
        '192.168.1.120',
        '192.168.1.121',
        '192.168.1.122',
        '192.168.1.123',
        '192.168.1.124',
        '192.168.1.125',
        '192.168.1.126',
        '192.168.1.127',
        '192.168.1.129',
        '192.168.1.130',
        '192.168.1.131',
        '192.168.1.132',
        '192.168.1.133',
        '192.168.1.134',
        '192.168.1.135',
        '192.168.1.136',
        '192.168.1.137',
        '192.168.1.138',
        '192.168.1.139',
        '192.168.1.141',
        '192.168.1.142',
        '192.168.1.143',
        '192.168.1.144',
        '192.168.1.145',
        '192.168.1.146',
        '192.168.1.147',
        '192.168.1.148',
        '192.168.1.149',
        '192.168.1.150',
        '192.168.1.151',
        '192.168.1.152',
    ]
    victim_ip = '192.168.1.128'
    middle_node_ip = ''
    node_number = len(ip_list)
    sleep_time = 0.07
    start_min = 10
    end_min = start_min + duration
    active_time = 5
    attack_start = 0
    duration_delta = timedelta(seconds=duration * 60)
    # send boot up information to nodes
    for ip in ip_list:
        if ip == victim_ip:
            continue
        # ip, active_time, sleep, k, is_ddos, is_end, victim_ip, clk
        # active_control_client(ip, active_time, sleep_time, k, False, False, victim_ip, 0, node_number, attack_ratio,
        #                       attack_start, duration_delta, ip_list)
    print('Start Tracking')
    treads_dic = dict()

    print('Start Experiment')

    # active_time, start_min, end_min, prec, k, sleep, victim_ip
    node_time_control(active_time, start_min, end_min, attack_ratio, k, sleep_time, victim_ip, node_number,
                      attack_start, duration_delta, ip_list)

    for key in treads_dic:
        treads_dic[key].join()

    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')
    print('Ending Time', formatted_time)
    print('Experiments End')
    final_data_processing(victim_ip, ip_list, k, attack_ratio, active_time, duration, memo)
    print('Data Integration Ends')
    time.sleep(240)


def node_time_control(active_time, start_min, end_min, prec, k, sleep, victim_ip, node_number, attack_start,
                      duration, ip_list):
    act_dic = dict()
    folder_path = "./dataFile"  # 请替换为你的文件夹路径
    all_files = os.listdir(folder_path)  # 获取文件夹下的所有文件名
    file_path_list = []  # 创建一个空列表来存储.csv文件的文件名
    for file in all_files:
        if file.endswith('.csv'):
            file_path_list.append('./dataFile/{}'.format(file))
    start_min = start_min * 60 / active_time
    end_min = end_min * 60 / active_time
    # for path in file_list:
    #     file_path_list.append(file_path.format(path))
    nums = random.sample(range(len(ip_list)), int(prec * len(ip_list)))
    selected_nodes = []
    for i in nums:
        if ip_list[i] != victim_ip:
            selected_nodes.append(ip_list[i])
    print('Selected Botnet: ', selected_nodes)
    i = 0
    for node in ip_list:
        data_frame = pd.read_csv(file_path_list[i])
        act_list = data_frame['ACTIVE'].to_list()
        act_dic[node] = act_list
        i += 1
    # print(act_dic)
    
    # record data for inference reference
    with open('./inference/inference_ref.csv', 'w') as file:
        field = ['time']
        for ip in ip_list:
            field.append(int(ip.split(".")[-1]))
        writer = csv.DictWriter(file,
                                    fieldnames=field)
        writer.writeheader()

    start_time = time.time()
    clk = 0
    with open('./inference/inference_ref.csv', 'a') as file:
        while True:
            if clk > 438:
                break
            clk += 1
            inf_data = {'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}
            for ip in ip_list:
                inf_data[int(ip.split(".")[-1])] = 0
            if start_min <= clk < end_min:
                print('attack start', clk, datetime.now())
                if attack_start == 0:
                    attack_start = datetime.now()
                ddos_attack_control(ip_list, prec, active_time, sleep, k, victim_ip, clk, node_number, act_dic,
                                    selected_nodes, attack_start, duration)
                for node in ip_list:
                    inf_data[int(node.split(".")[-1])] = 1
            else:
                for node in ip_list:
                    if act_dic[node][clk] == 1:
                        # active_control_client(node, active_time, sleep, 0, False, False, victim_ip, clk, node_number, prec,
                        #                     attack_start, duration, ip_list)
                        # node.cmd('sudo ./benign_behavior.py --time {} --n 10 --sleep 0.1 &'.format(time_interval))
                        print(node, ' is active.')
                print(clk)
            writer = csv.DictWriter(file,
                                    fieldnames=field)
            writer.writerow(inf_data)
            print('----------------------------')
            time.sleep(active_time)
    for node in ip_list:
        print('Sending End Signal to {}'.format(node))
        active_control_client(node, active_time, sleep, 0, False, True, victim_ip, clk, node_number, prec,
                              attack_start, duration, ip_list)


def active_control_client(ip, active_time, sleep, k, is_ddos, is_end, victim_ip, clk, node_number, attack_ratio,
                          attack_start, duration, ip_list):
    c_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # arr = [active_time, sleep, k, is_ddos, is_end]
    data_dic = {'active_time': active_time,
                'sleep': sleep,
                'k': k,
                'is_ddos': is_ddos,
                'is_end': is_end,
                'victim_ip': victim_ip,
                'clk': clk,
                'node_ip': ip,
                'node_number': node_number,
                'attack_ratio': attack_ratio,
                'attack_start': attack_start,
                'duration': duration,
                'ip_list': ip_list
                }
    c_socket.sendto(pickle.dumps(data_dic), (ip, 9565))


def ddos_attack_control(ip_list, perc, active_time, sleep, k, victim_ip, clk, node_number, act_dic, selected_nodes,
                        attack_start, duration):
    for node in selected_nodes:
        if node != victim_ip:
            active_control_client(node, active_time, sleep, k, True, False, victim_ip, clk, node_number, perc,
                                  attack_start, duration, ip_list)
    for node in ip_list:
        if node not in selected_nodes and act_dic[node][clk] == 1:
            print(node, ' is active.')
            active_control_client(node, active_time, sleep, 0, False, False, victim_ip, clk, node_number, perc,
                                  attack_start, duration, ip_list)
    # if '192.168.1.158' in selected_nodes:
    #     print('start attack')
    #     t = threading.Thread(target=ddos_attack_simulation.ddos_attack, args=(victim_ip, active_time, k))
    #     t.start()


def file_transferring_and_generating(host, port, username, password, remote_file_path, local_file_path):
    transport = paramiko.Transport((host, port))
    transport.connect(username=username, password=password)

    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.get(remote_file_path, local_file_path)

    sftp.close()
    transport.close()


def get_ip_address():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip_address = sock.getsockname()[0]
        return ip_address
    except socket.error:
        return None
    finally:
        sock.close()


def final_data_processing(victim_ip, ip_list, k, ratio, active_time, duration, memo):
    port = 22
    username = ''
    password = ''
    remote_file_path = '/home/pi/Documents/MiniTest/dataFile/packet_volume/packet_volume_info_{}.csv'
    local_file_path = '/home/pi/Documents/MiniTest/dataFile/final_data/packet_volume_info_{}.csv'
    final_data = pd.DataFrame()
    recv_ip = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_ip = get_ip_address()
    sock.bind((ctrl_ip, 9568))
    while True:
        if len(recv_ip) >= len(ip_list):
            break
        data, address = sock.recvfrom(1024)
        received_data = pickle.loads(data)
        if address[0] not in recv_ip and received_data[0]:
            recv_ip.append(address[0])
            print('receive ending signal from :', address[0])
            recv_ip.sort()
            print(set(ip_list)-set(recv_ip), ' not received.')
    print('All nodes finished processing data, start requesting...')
    for host in ip_list:
        if host == victim_ip:
            continue
        print('Processing data from', host)
        try:
            file_transferring_and_generating(host, port, username, password, remote_file_path.format(host[-3:]),
                                             local_file_path.format(host[-3:]))
        except FileNotFoundError:
            print('No such file')
            continue
        except Exception as e:
            print(f"Error: {e}")
            continue
        df = pd.read_csv(local_file_path.format(host[-3:]))
        final_data = pd.concat([final_data, df], ignore_index=True)
    final_data.to_csv('/home/pi/Documents/MiniTest/dataFile/final_data/{}_{}_{}_{}_data.csv'.format(k, ratio, duration,
                                                                                                    memo), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=float, default=1)
    parser.add_argument("--name", type=str, default='traffic')
    args = parser.parse_args()
    k_para = [8]
    Ratio = [1]
    duration = [10, 20]
    for k in k_para:
        for perc in Ratio:
            for t in duration:
                ddos_experiment(args.t, args.name, k, perc, t, 'training')
                time.sleep(120)
                ddos_experiment(args.t, args.name, k, perc, t, 'testing')
                time.sleep(120)
                ddos_experiment(args.t, args.name, k, perc, t, 'validation')
