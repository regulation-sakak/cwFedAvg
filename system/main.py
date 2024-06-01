# cwFedAvg

import copy
import argparse
import os
import json
import time
import warnings
import numpy as np
import logging

from flcore.servers.serveravgCW import cwFedAvg
from flcore.trainmodel.models import *

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

def run(args):
    time_list = []
    model_str = args.model

    for i in range(args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        if model_str == "cnn":
            if "mnist" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        if args.algorithm == "cwFedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = cwFedAvg(args, i)
        else:
            raise NotImplementedError

        server.train()
        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-gr', "--global_rounds", type=int, default=1000)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-algo', "--algorithm", type=str, default="cwFedAvg")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-cw', "--add_cw", action='store_true')
    parser.add_argument('-wdr', "--add_wdr", action='store_true')
    parser.add_argument('-gt', "--use_true_dist", action='store_true')
    parser.add_argument('-wd', "--weight_decay", type=float, default=10.0)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    args.device = "cuda"

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), 'dataset')
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    file_path = os.path.join(dataset_path, 'config.json')
    with open(file_path, 'r') as file:
        data = json.load(file)
        args.num_classes = data['num_classes']
        args.dataset_type = data['partition']
        data_statistic = data['Size of samples for labels in clients']

    data_dist = np.zeros((args.num_clients, args.num_classes))
    for i in range(args.num_clients):
        client_data_dist_np = np.array(data_statistic[i])
        data_dist[i, client_data_dist_np[:, 0]] = client_data_dist_np[:, 1]

    args.data_dist = data_dist.tolist()

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Partition: {}".format(args.dataset_type))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("=" * 50)

    run(args)
