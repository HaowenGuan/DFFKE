import copy
import random
import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
import os


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.n_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

        model = load_item(self.clients[0].role, 'model', self.clients[0].save_folder_name)
        save_item(model, self.role, 'model', self.save_folder_name)

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            self.send_parameters()

            download_cost = 0
            _, size = load_item('Server', 'model', self.save_folder_name, get_size=True)
            download_cost += size * len(self.selected_clients)
            self.download_cost_list.append(download_cost)
            print(f"Download cost: {download_cost:.2f} MB")

            print(f"\n-------------Round number: {i}-------------")
            if i % self.eval_gap == 0:
                print("\nEvaluate heterogeneous models after FedAvg")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_ids()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc],
                                                   auto_break_patient=self.auto_break_patient):
                break

        print("\nBest accuracy.")
        print(f'{max(self.rs_test_acc):.2f}')
        print("Average time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        import torch
        num_of_round_reach_best_acc = torch.topk(torch.tensor(self.rs_test_acc), 1).indices[0] * self.eval_gap
        print(f"Total Upload cost: {sum(self.upload_cost_list[:num_of_round_reach_best_acc]):.2f} MB")
        print(f"Total Download cost: {sum(self.download_cost_list[:num_of_round_reach_best_acc]):.2f} MB")

        self.save_results()

    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        client = self.clients[self.uploaded_ids[0]]
        model = load_item(client.role, 'model', client.save_folder_name)
        for param in model.parameters():
            param.data.zero_()

        upload_cost = 0
        for w, cid in zip(self.uploaded_weights, self.uploaded_ids):
            client = self.clients[cid]
            client_model, size = load_item(client.role, 'model', client.save_folder_name, get_size=True)
            upload_cost += size
            for server_param, client_param in zip(model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * w

        print(f"Upload cost: {upload_cost:.2f} MB")
        self.upload_cost_list.append(upload_cost)

        save_item(model, self.role, 'model', self.save_folder_name)

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        ids = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            print(f'Client {c.id}: Acc: {ct * 1.0 / ns}, AUC: {auc}')
            tot_auc.append(auc * ns)
            num_samples.append(ns)
            ids.append(c.id)

            print('Since all clients shares the same model and test set.')
            print('Skip testing rest of clients.')
            break

        return ids, num_samples, tot_correct, tot_auc
