#!/usr/bin/env python3

import sys
from datetime import datetime
import time
import socket
import csv
import argparse
sys.path.append("./inference/source_code/nn_training/")
sys.path.append("./inference/source_code/")

from inference.source_code.nn_training.generate_results import load_model_weights, load_dataset, \
    get_dataset_input_output, find_best_threshold


threshold_dic = {
  160: 0.6, 159: 0.5, 166: 0.8, 157: 0.5, 187: 0.6, 198: 0.5, 108: 0.7, 109: 0.7,
  110: 0.7, 111: 0.7, 112: 0.2, 113: 0.7, 114: 0.5, 115: 0.6, 116: 0.5, 117: 0.6,
  118: 0.5, 119: 0.5, 120: 0.7, 121: 0.7, 122: 0.7, 123: 0.6, 124: 0.7, 125: 0.6,
  126: 0.5, 127: 0.3, 129: 0.7, 130: 0.2, 131: 0.4, 132: 0.7, 133: 0.6, 134: 0.7,
  135: 0.4, 136: 0.7, 137: 0.5, 138: 0.4, 139: 0.8, 141: 0.3, 142: 0.7, 143: 0.6,
  144: 0.6, 145: 0.8, 146: 0.7, 147: 0.8, 148: 0.8, 149: 0.7, 150: 0.5, 151: 0.6,
  152: 0.9
}


def load_the_data(data_path, model_path, sleep_time, node_number):
    # node_number = get_node_number()
    model, scaler, selected_nodes_for_correlation = load_model(model_path)
    # selected_nodes_for_correlation.append(140)
    print(selected_nodes_for_correlation)
    while True:
        try:
            data = load_dataset(data_path)
            break
        except Exception as e:
            print("Waiting for data for inference...")
            time.sleep(1)
    # threshold = find_best_threshold(
    #     model,
    #     "",
    #     "multiple_models_with_correlation",
    #     node_number,
    #     data,
    #     selected_nodes_for_correlation,
    #     1,
    #     10,
    #     scaler,
    #     False,
    #     True
    # )
    threshold = threshold_dic[node_number]
    avg_pred = []
    avg_whole = []
    with open('/home/pi/Documents/MiniTest/inference/{}_prediction_data.csv'.format(node_number), 'w') as file:
            writer = csv.DictWriter(file,
                                    fieldnames=['time', 'prediction'])
            writer.writeheader()
    while True:
        try:
            data = load_dataset(data_path)
            X_out, y_out, df_out, temp_columns = get_dataset_input_output(
                "", "multiple_models_with_correlation",
                data,
                selected_nodes_for_correlation,
                1,
                10,
                scaler,
                False,
                True
            )
        except Exception as e:
            print(f"Error in getting dataset input and output: {e}")
            continue

        t2 = datetime.now()
        
        if X_out.shape[1:] != model.input_shape[1:]:
            print(X_out, model.input_shape[1:])
            print("Skipping due to incompatible shape of X_out.")
            continue
        
        try:
            test_predictions_baseline = model.predict(X_out)
        except Exception as e:
            print(f"Error in model prediction: {e}")
            continue
        write_data = {'time': datetime.now(), 'prediction': 0}
        if test_predictions_baseline >= threshold:
            write_data['prediction'] = 1
            print('DDoS Botnet!')
        else:
            print('Benign')
            
        with open('/home/pi/Documents/MiniTest/inference/{}_prediction_data.csv'.format(node_number), 'a') as file:
            writer = csv.DictWriter(file,
                                    fieldnames=['time', 'prediction'])
            writer.writerow(write_data)
        t3 = datetime.now()
        # avg_pred.append((t3-t2).total_seconds())
        # avg_whole.append((t3-t1).total_seconds())
        # print("start time: ", t1)
        # print("time to read data: ", t2-t1)
        # print("time just for prediction: ", t3-t2, " average: ", sum(avg_pred)/len(avg_pred))
        # print("time for the whole process: ", t3-t1, " average: ", sum(avg_whole)/len(avg_whole))
        # print("time getting the result: ", t3)
        
        print(test_predictions_baseline)
        time.sleep(3)


def load_model(model_path):
    model, scaler, selected_nodes_for_correlation = load_model_weights(
        model_path,
        139,
        "val_binary_accuracy",
        "max"
    )
    return model, scaler, selected_nodes_for_correlation


def get_node_number():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip_address = sock.getsockname()[0]
        return int(ip_address.split(".")[-1])
    except socket.error:
        return None
    finally:
        sock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='138')
    args = parser.parse_args()
    load_the_data("./dataFile/data_for_inference/data_for_inference.csv", "./model/", 5, args.model)
