import time
import numpy as np
from flcore.clients.clientproto import clientProto
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict


class FedProto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientProto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.n_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            print(f"\n-------------Round number: {i}-------------")

            for client in self.selected_clients:
                client.train()

            if i > 1:
                _, download_cost = load_item('Server', 'global_protos', self.save_folder_name, get_size=True)
                download_cost *= len(self.selected_clients)
                print(f"Download cost: {download_cost:.2f} MB")
                self.download_cost_list.append(download_cost)

            if i % self.eval_gap == 0:
                print("\nEvaluate heterogeneous models after Local Training")
                self.evaluate()

            self.receive_protos()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], auto_break_patient=self.auto_break_patient):
                break

        print("\nBest accuracy.")
        print(f'{max(self.rs_test_acc):.2f}')
        print("Average time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        import torch
        num_of_round_reach_best_acc = torch.topk(torch.tensor(self.rs_test_acc), 1).indices[0] * self.eval_gap
        print(f"Total Upload cost: {sum(self.upload_cost_list[:num_of_round_reach_best_acc]):.2f} MB")
        print(f"Total Download cost: {sum(self.download_cost_list[:num_of_round_reach_best_acc]):.2f} MB")

        self.save_results()
        

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        uploaded_protos = []
        upload_cost = 0
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos, size = load_item(client.role, 'protos', client.save_folder_name, get_size=True)
            upload_cost += size
            uploaded_protos.append(protos)

        print(f"Upload cost: {upload_cost:.2f} MB")
        self.upload_cost_list.append(upload_cost)
            
        global_protos = proto_aggregation(uploaded_protos)
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)
    

# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label