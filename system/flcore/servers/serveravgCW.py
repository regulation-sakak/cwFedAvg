import time
import random
import copy
import numpy as np
from flcore.clients.clientavgCW import cwclientAVG
from flcore.servers.serverbase import Server


class cwFedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(cwclientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            assert (len(self.clients) > 0)

            for client in self.clients:
                start_time = time.time()

                if self.args.add_cw:
                    client.local_initializtion_cw(self.cw_global_model)
                else:
                    client.set_parameters(self.global_model)

                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)


            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                if i == 0:
                    self.args.batch_eval = True
                else:
                    self.args.batch_eval = False
                client.train()

            if self.args.add_cw:
                self.receive_models_cw()
            else:
                self.receive_models()

            if self.args.add_cw:
                self.aggregate_parameters_cw()
            else:
                self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

    def send_models_cw(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.local_initialization_cw(self.our_global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models_cw(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_weights_cw = []
        self.uploaded_models = []
        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)
            self.uploaded_weights.append(client.train_samples)
            if self.args.use_true_dist:
                weight_list = client.data_dist
            else:
                weight_list = client.aggregate_weight_calc()
            weight_list = [x / sum(weight_list) for x in weight_list]
            self.uploaded_weights_cw.append(weight_list)
        # store client weight of FedAVG
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters_cw(self):
        assert (len(self.uploaded_models) > 0)

        for ig in range(len(self.cw_global_model)):
            self.cw_global_model[ig] = copy.deepcopy(self.uploaded_models[0])
            for param in self.cw_global_model[ig].parameters():
                param.data.zero_()

        fedavg_weight = np.array(self.uploaded_weights).reshape(self.num_clients, 1)
        fedavg_weight = np.repeat(fedavg_weight, self.num_classes, axis=1)
        uploaded_weight_np = np.array(self.uploaded_weights_cw)
        uploaded_weight_np = uploaded_weight_np * fedavg_weight

        marginal_weight_np = np.tile(np.sum(uploaded_weight_np, axis=0), (self.num_clients, 1))
        normalized_weight = np.divide(uploaded_weight_np, marginal_weight_np).tolist()

        for idx in range(len(normalized_weight)):
            w = normalized_weight[idx]
            local_model = self.uploaded_models[idx]
            for idg in range(len(self.cw_global_model)):
                for target, source in zip(self.cw_global_model[idg].parameters(),
                                          local_model.parameters()):
                    target.data += source.data.clone() * w[idg]
