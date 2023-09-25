#!/usr/bin/env python3
import json
import socket
import csv
import copy
from datetime import datetime, timedelta


# global var
record_for_inf = []
packet_volume_info = dict()
pre_time = datetime.now()
hour_data = -1
total_record_number = 0


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


def udp_server(host, port, active_time, start_time, node_num, attack_ratio, attack_start,
               duration, k, ip_list, victim_ip):
    global packet_volume_info, record_for_inf
    for ip in ip_list:
        packet_volume_info[ip] = 0
    s_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s_socket.bind((host, port))
    path = './dataFile/packet_volume/{}.csv'.format(host)
    print('UDP server listening on {}:{}'.format(host, port))
    # check whether the target file exist
    try:
        with open(path, 'w') as file:
            writer = csv.DictWriter(file,
                                    fieldnames=['data', 'address', 'time'])
            writer.writeheader()
    except FileExistsError:
        pass

    # write header for inference
    header_info = ['BEGIN_DATE',
                   'END_DATE',
                   'NUM_NODES',
                   'ATTACK_RATIO',
                   'ATTACK_START_TIME',
                   'ATTACK_DURATION',
                   'ATTACK_PARAMETER',
                   'NODE',
                   'LAT',
                   'LNG',
                   'TIME',
                   'TIME_FEATURE',
                   'ACTIVE',
                   'PACKET',
                   'ATTACKED']
    if victim_ip in ip_list:
        ip_list.remove(victim_ip)
    for node in ip_list:
        header_info.append('PACKET_{}'.format(node[-3:]))
        header_info.append('NODE_{}'.format(node[-3:]))
    try:
        with open('./dataFile/data_for_inference/data_for_inference.csv', 'w') as file:
            writer = csv.DictWriter(file,
                                    fieldnames=header_info)
            writer.writeheader()
    except FileExistsError:
        pass

    # Udp Server Listen
    while True:
        data, address = s_socket.recvfrom(1024)
        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-5]

        packet_info = {
            'data': data.decode(),
            'address': address[0],
            'time': formatted_time
        }
        record_for_inference(host, packet_info, active_time, start_time, node_num, attack_ratio, attack_start,
                             duration, k, ip_list, header_info)
        # print('Received data from {} : {}'.format(address, data.decode()))
        with open(path, 'a') as file:
            writer = csv.DictWriter(file,
                                    fieldnames=['data', 'address', 'time'])
            writer.writerow(packet_info)


def record_for_inference(host, packet_info, active_time, start_time, node_num, attack_ratio, attack_start,
                         duration, k, ip_list, header_info):
    global record_for_inf, packet_volume_info, pre_time, hour_data, total_record_number
    if datetime.now() - pre_time >= timedelta(seconds=active_time):
        if total_record_number % 6 == 0:
            hour_data = (hour_data + 1) % 24
        if len(record_for_inf) >= 10:
            record_for_inf.pop(0)
        active = 0
        attacked = 0
        try:
            with open('./shared_variable/shared_variable.json', 'r') as file:
                json_data = json.load(file)
                if json_data['ACTIVE']:
                    active = 1
                if json_data['ATTACKED']:
                    attacked = 1
        except json.JSONDecodeError:
            print("Error: Unable to decode JSON data.")
        except FileNotFoundError:
            print("Error: File not found.")
        except Exception as e:
            print(f"Error: {e}")

        record_info = {
            'BEGIN_DATE': start_time.strftime("%Y-%m-%d"),
            'END_DATE': start_time.strftime("%Y-%m-%d"),
            'NUM_NODES': node_num,
            'ATTACK_RATIO': attack_ratio,
            'ATTACK_START_TIME': attack_start,
            'ATTACK_DURATION': duration,
            'ATTACK_PARAMETER': k,
            'NODE': host[-3:],
            'LAT': 50.46388319,
            'LNG': 35.37576942,
            'TIME': pre_time,
            'TIME_FEATURE': hour_data,
            'ACTIVE': active,
            'PACKET': packet_volume_info[host],
            'ATTACKED': attacked
        }
        for node in ip_list:
            if node == host:
                record_info['NODE_{}'.format(node[-3:])] = 1
                record_info['PACKET_{}'.format(node[-3:])] = packet_volume_info[host]
                continue
            record_info['NODE_{}'.format(node[-3:])] = 0
            record_info['PACKET_{}'.format(node[-3:])] = packet_volume_info[node]
        record_for_inf.append(record_info)
        total_record_number += 1
        for key in packet_volume_info:
            packet_volume_info[key] = 0
        pre_time = datetime.now()
        with open('./dataFile/data_for_inference/data_for_inference.csv', 'w') as file:
            writer = csv.DictWriter(file,
                                    fieldnames=header_info)
            writer.writeheader()
            if len(record_for_inf) < 10 and len(record_for_inf) != 0:
                tmp_data = copy.copy(record_for_inf[len(record_for_inf)-1])
                record_for_inf = []
                for i in range(10):
                    record_for_inf.append(copy.copy(tmp_data))
            for data in record_for_inf:
                writer.writerow(data)
    packet_volume_info[packet_info['address']] += int(packet_info['data'])


if __name__ == '__main__':
    node_ip = get_ip_address()
    node_port = 9564
    print('{}:9564'.format(node_ip))
    # udp_server(node_ip, node_port)
