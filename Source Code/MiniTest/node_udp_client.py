#!/usr/bin/env python3
import argparse
import socket
from scapy.all import *
import time
import pcapy
from scapy.layers.inet import TCP, UDP, IP


# callback function
def send_volume(packet_count, ip_list):
    c_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for target_ip in ip_list:
        c_socket.sendto(str(packet_count).encode('utf-8'), (target_ip, 9564))


def client(int_time, ip_list):
    current = time.time()
    capture_session = pcapy.open_live('any', 65536, True, 1)

    packet_count = 0
    while True:
        if time.time() - current > int_time:
            send_volume(packet_count, ip_list)
            packet_count = 0
            current = time.time()
        _, packet = capture_session.next()
        packet_count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=float, default=1)
    args = parser.parse_args()
    client(args.t)


