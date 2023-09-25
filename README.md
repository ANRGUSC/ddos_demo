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

## Training neural network and generate performance results

The LSTM/MM-WC model extends from our previous work, *Correlation Aware DDoS Detection In IoT Systems*, The codebase for model training is publicly available in the repository [here](https://github.com/ANRGUSC/correlation_aware_ddos_iot). A pre-print version of the original paper can be accessed [here](https://arxiv.org/abs/2302.07982) .

The code necessary for training the LSTM/MM-WC model is located in the folder [source_code](/Source Code/MiniTest/inference/source_code). For further details on model training, please consult the repository documentation.