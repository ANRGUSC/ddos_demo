#!/usr/bin/env python3

import argparse
import pcapy
from struct import *
from datetime import datetime

import csv

from scapy.layers.l2 import Ether
from scapy.layers.inet import TCP, UDP, IP


def packet_capture(dev, file_name):
    # Decrease the buffer size
    pc = pcapy.open_live(dev, 2048, 1, 10)
    print("open succeeded")

    path = './dataFile/output_data/{}.csv'.format(file_name)
    prev_packet_time = None

    # Open file once at the start
    try:
        file = open(path, 'w')
        writer = csv.DictWriter(file,
                                fieldnames=['packet_length', 'packet_frame_len', 'source_ip', 'destination_ip',
                                            'source_port', 'destination_port', 'received_time', 'time_difference',
                                            'ttl', 'protocol'])
        writer.writeheader()
    except FileExistsError:
        pass
    

    while True:
        try:
            (header, packet) = pc.next()
            parsed_packet = Ether(packet)

            if IP in parsed_packet:
                ip_layer = parsed_packet[IP]
                current_packet_time = header.getts()[0]
                time_diff = current_packet_time - prev_packet_time if prev_packet_time is not None else None

                packet_info = {
                    'packet_length': len(packet),
                    'packet_frame_len': len(parsed_packet),
                    'source_ip': ip_layer.src,
                    'destination_ip': ip_layer.dst,
                    'source_port': ip_layer.sport if TCP in ip_layer or UDP in ip_layer else None,
                    'destination_port': ip_layer.dport if TCP in ip_layer or UDP in ip_layer else None,
                    'received_time': datetime.fromtimestamp(current_packet_time).strftime('%Y-%m-%d %H:%M:%S'),
                    'time_difference': time_diff,
                    'ttl': ip_layer.ttl,
                    'protocol': ip_layer.proto
                }
                # print(packet_info)
                writer.writerow(packet_info)

                prev_packet_time = current_packet_time

        except pcapy.PcapError as e:
            print("PcapError:", e)
            continue

        except Exception as e:
            print("Error:", e)
            continue

    # Close file at the end
    file.close()


def network_capture(node):
    packet_capture('wlan0', node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=str, default='10.0.0.0')
    args = parser.parse_args()
    network_capture(args.node)
