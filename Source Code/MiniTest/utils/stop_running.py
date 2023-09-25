#!/usr/bin/env python3
import pickle
import socket
import time
from datetime import timedelta


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

for i in range(1):
    victim_ip = '192.168.1.128'
    middle_node_ip = ''
    node_number = len(ip_list)
    sleep_time = 0.07
    start_min = 10
    end_min = start_min + 0.5
    active_time = 5
    attack_start = 0
    duration_delta = timedelta(seconds=0.5 * 60)
    # send boot up information to nodes
    for ip in ip_list:
        if ip == victim_ip:
            continue
        # ip, active_time, sleep, k, is_ddos, is_end, victim_ip, clk
        active_control_client(ip, active_time, sleep_time, 10, False, True, victim_ip, 0, node_number, 1,
                              attack_start, duration_delta, ip_list)
    print(i)
    time.sleep(10)