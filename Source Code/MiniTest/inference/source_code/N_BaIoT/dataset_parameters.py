"""Define the parameters of N-BaIoT paper
"""

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
}  # , "Channel Jitter": "HH_jit"
time_windows = {
    "100-ms": "L5",
    "500-ms": "L3",
    "1.5-sec": "L1",
    "10-sec": "L0.1",
    "1-min": "L0.01",
}
statistics = {"Number": "weight"}
