# Correlation-aware Ubiquitous Detection of DDoS in IoT Systems

This repository presents the source code for the paper __*Demo Abstract- CUDDoS: Correlation-aware Ubiquitous Detection of DDoS in IoT Systems*__.

## Instruction for running the code

Executing the code requires sudo access on both the Client Node and the Control Server. After cloning the repository, navigate to the respective directory and install the necessary libraries to run the code.

~~~bash
cd ./MiniTest
sudo ./install_lib.sh
~~~

On the client IoT node, execute the following command prior to initiating the Control Server.

~~~bash
sudo ./main_client.py
~~~

Once the client IoT node is operational, proceed to execute the corresponding command on the Control Server side.

~~~bash
sudo ./main.py
~~~

