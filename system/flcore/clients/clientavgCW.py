import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class cwclientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.aggregate_global = copy.deepcopy(args.model)
        self.weight_decay = self.args.weight_decay
        self.data_dist = args.data_dist[id]
        gt = [element / sum(self.data_dist) for element in self.data_dist]
        self.gt = torch.tensor(gt).to(self.device)

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()

        start_time = time.time()

        for epoch in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                if self.args.add_wdr:
                    fc_weight_norm = torch.norm(self.model.head.weight, dim=1).unsqueeze(0)
                    fc_weight = fc_weight_norm / torch.sum(fc_weight_norm)
                    wd_regularizer = torch.norm(fc_weight - self.gt, p=2)
                    loss += 0.5 * self.weight_decay * wd_regularizer

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def aggregate_weight_calc(self):
        fc_weight_norm = torch.norm(self.model.head.weight, dim=1).unsqueeze(0)
        fc_weight_norm_list = fc_weight_norm.detach().cpu().numpy().tolist()[0]
        return fc_weight_norm_list

    def local_initializtion_cw(self, received_cw_global_models):
        if self.args.use_true_dist:
            weight_list = self.data_dist
        else:
            weight_list = self.aggregate_weight_calc()
        weight_list = [element / sum(weight_list) for element in weight_list]

        self.aggregate_global = copy.deepcopy(received_cw_global_models[0])
        for param in self.aggregate_global.parameters():
            param.data = torch.zeros_like(param.data)

        for w, global_model in zip(weight_list, received_cw_global_models):
            for agg_param, global_param in zip(self.aggregate_global.parameters(), global_model.parameters()):
                agg_param.data += global_param.data.clone() * w

        for new_param, old_param in zip(self.aggregate_global.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
