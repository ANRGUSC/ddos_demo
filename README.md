# Correlation-aware Ubiquitous Detection of DDoS in IoT Systems

This repository presents the source code for the paper __*Demo Abstract- CUDDoS: Correlation-aware Ubiquitous Detection of DDoS in IoT Systems*__.

## Instruction for running the code

Executing the code requires sudo access on both the Client Node and the Control Server. After cloning the repository, navigate to the respective directory and install the necessary libraries to run the code.

~~~bash
cd ./MiniTest
sudo pip3 install -r requirements.txt
~~~

If you fail to install scapy or pcapy, please run the following command.

~~~bash
sudo ./install_lib.sh
~~~

It should be noted that the Tensorflow and shap are required but are not included in the *requirements.txt*, and it is required to run the inference script on the Client nodes. The version information is shown below:

~~~bash
virtualenv==20.24.2
shap==0.42.1
tensorflow==2.13.0
~~~

## Real time traffic generation

On the client IoT node, execute the following command prior to initiating the Control Server.

~~~bash
sudo ./main_client.py
~~~

Once the client IoT node is operational, proceed to execute the corresponding command on the Control Server side.

~~~bash
sudo ./main.py
~~~

## DDoS detection

To detect botnet with our LSTM/MM-WC model, please use the following command to run the *inference.py*:

~~~bash
source tftest/bin/activate
cd ./MiniTest
./inference.py
~~~

Before executing the script, ensure that the requisite model is located in the folder [/Source Code/MiniTest/model](https://github.com/ANRGUSC/ddos_demo/tree/main/Source%20Code/MiniTest/model). By default, model 138 is employed. Should you wish to utilize alternative models, execute the following command:
~~~bash
./inference.py --model {}
~~~

Please make sure that the tensorflow environment is correctly installed and activated before running the inference script.

## Training neural network and generate performance results

The LSTM/MM-WC model extends from our previous work, *Correlation Aware DDoS Detection In IoT Systems*, The codebase for model training is publicly available in the repository [here](https://github.com/ANRGUSC/correlation_aware_ddos_iot). A pre-print version of the original paper can be accessed [here](https://arxiv.org/abs/2302.07982) .

The code necessary for training the LSTM/MM-WC model is located in the folder [/Source Code/MiniTest/inference/source_code](https://github.com/ANRGUSC/ddos_demo/tree/main/Source%20Code/MiniTest/inference/source_code). For further details on model training, please consult the repository documentation in our [prior work](https://github.com/ANRGUSC/correlation_aware_ddos_iot).