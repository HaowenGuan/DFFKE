import copy
import random
import time
from flcore.clients.clientlg import clientLG
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread


class LG_FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientLG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.n_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.auto_break_patient *= 2  # Due to Testing twice each round

        head = load_item(self.clients[0].role, 'model', self.clients[0].save_folder_name).head
        save_item(head, self.role, 'head', self.save_folder_name)


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            self.send_parameters()

            print(f"\n-------------Round number: {i}-------------")
            if i % self.eval_gap == 0:
                print("\nEvaluate heterogeneous models after LG_FedAvg")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            if i % self.eval_gap == 0:
                print("\nEvaluate heterogeneous models After local training")
                self.evaluate()

            self.receive_ids()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], auto_break_patient=self.auto_break_patient):
                break

        print("\nBest accuracy.")
        print(f'{max(self.rs_test_acc):.2f}')
        print("Average time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()


    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        client = self.clients[self.uploaded_ids[0]]
        head = load_item(client.role, 'model', client.save_folder_name).head
        for param in head.parameters():
            param.data.zero_()
            
        for w, cid in zip(self.uploaded_weights, self.uploaded_ids):
            client = self.clients[cid]
            client_head = load_item(client.role, 'model', client.save_folder_name).head
            for server_param, client_param in zip(head.parameters(), client_head.parameters()):
                server_param.data += client_param.data.clone() * w

        save_item(head, self.role, 'head', self.save_folder_name)
        