#!/usr/bin/env python3
import csv
from collections import defaultdict
from datetime import datetime, timedelta


def data_processing(node_ip, start_time, node_num, attack_ratio, attack_start, end_time, duration, k, active_list,
                    ddos_list, active_time, victim_ip, ip_list):
    data_info = ['BEGIN_DATE',
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

    # file_list = ['1152', '13101', '23093', '25668', '27068', '27638', '28381', '31867', '31973', '33925']
    if victim_ip in ip_list:
        ip_list.remove(victim_ip)
    for node in ip_list:
        data_info.append('PACKET_{}'.format(node[-3:]))
        data_info.append('NODE_{}'.format(node[-3:]))

    try:
        with open('./dataFile/packet_volume/packet_volume_info_{}.csv'.format(node_ip[-3:]), 'w') as file:
            writer = csv.DictWriter(file,
                                    fieldnames=data_info)
            writer.writeheader()
    except FileExistsError:
        pass

    # read the packet volume info
    data_file_path = './dataFile/packet_volume/{}.csv'.format(node_ip)
    data_length = 0

    with open(data_file_path, mode='r') as data_file:
        packet_volume_info = defaultdict(list)
        csv_reader = csv.DictReader(data_file)
        for row in csv_reader:
            address = row['address']
            packet_volume_info[address].append(row)
            data_length += 1
    # sample:
    # data_by_address = {
    #     "192.168.1.109": [
    #         {"data": "1", "address": "192.168.1.109", "time": "2023-07-03 23:42:31"},
    #         {"data": "163", "address": "192.168.1.109", "time": "2023-07-03 23:42:36"},
    #     ]
    # }

    # read node activity data and processing

    # write data
    with open('./dataFile/packet_volume/packet_volume_info_{}.csv'.format(node_ip[-3:]), 'a') as file:
        writer = csv.DictWriter(file,
                                fieldnames=data_info)
        hour_data = -1
        for i in range(432):
            if i % 6 == 0:
                hour_data = (hour_data + 1) % 24
            current = start_time + timedelta(seconds=active_time * i)
            # if current > end_time:
            #     break
            active = 0
            attacked = 0
            packet_num = 0
            for time in active_list:
                if current <= time < current + timedelta(seconds=active_time):
                    active = 1
                    break
            for time in ddos_list:
                if current <= time < current + timedelta(seconds=active_time):
                    attacked = 1
                    active = 1
                    break
            for data in packet_volume_info[node_ip]:
                if current < datetime.strptime(data['time'], "%Y-%m-%d %H:%M:%S.%f") <= current + timedelta(
                        seconds=active_time):
                    packet_num += int(data['data'])
                if datetime.strptime(data['time'], "%Y-%m-%d %H:%M:%S.%f") > current + timedelta(
                        seconds=active_time):
                    break

            packet_info = {
                'BEGIN_DATE': start_time.strftime("%Y-%m-%d"),
                'END_DATE': start_time.strftime("%Y-%m-%d"),
                'NUM_NODES': node_num,
                'ATTACK_RATIO': attack_ratio,
                'ATTACK_START_TIME': attack_start,
                'ATTACK_DURATION': duration,
                'ATTACK_PARAMETER': k,
                'NODE': node_ip[-3:],
                'LAT': 50.46388319,
                'LNG': 35.37576942,
                'TIME': current,
                'TIME_FEATURE': hour_data,
                'ACTIVE': active,
                'PACKET': packet_num,
                'ATTACKED': attacked
            }
            for node in ip_list:
                if node == node_ip:
                    packet_info['NODE_{}'.format(node[-3:])] = 1
                    packet_info['PACKET_{}'.format(node[-3:])] = packet_num
                    continue
                packet_info['NODE_{}'.format(node[-3:])] = 0
                packet_info['PACKET_{}'.format(node[-3:])] = 0
                for data in packet_volume_info[node]:
                    if current < datetime.strptime(data['time'], "%Y-%m-%d %H:%M:%S.%f") <= current + timedelta(
                            seconds=active_time):
                        packet_info['PACKET_{}'.format(node[-3:])] = int(data['data'])
                        break
            print(packet_info)
            writer.writerow(packet_info)
    print('End Write Data')


if __name__ == "__main__":
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-5]
    data_processing('192.168.1.159', '2023-07-07 00:25:38.1', 10, 1, '2023-07-07 00:25:38.1', '2023-07-07 00:25:38.1',
                    10, 1, [], [], 1, '')
