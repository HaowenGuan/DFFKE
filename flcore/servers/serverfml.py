import copy
import random
import time
from flcore.clients.clientfml import clientFML
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from models.model import BaseHeadSplit


class FML(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            global_model = BaseHeadSplit(args, 0).to(args.device)            
            save_item(global_model, self.role, 'global_model', self.save_folder_name)
        
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFML)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.n_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            print(f"\n-------------Round number: {i}-------------")

            for client in self.selected_clients:
                client.train()

            _, download_cost = load_item('Server', 'global_model', self.save_folder_name, get_size=True)
            download_cost *= len(self.selected_clients)
            print(f"Download cost: {download_cost:.2f} MB")
            self.download_cost_list.append(download_cost)

            if i % self.eval_gap == 0:
                print("\nEvaluate heterogeneous models after local training")
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
        import torch
        num_of_round_reach_best_acc = torch.topk(torch.tensor(self.rs_test_acc), 1).indices[0] * self.eval_gap
        print(f"Total Upload cost: {sum(self.upload_cost_list[:num_of_round_reach_best_acc]):.2f} MB")
        print(f"Total Download cost: {sum(self.download_cost_list[:num_of_round_reach_best_acc]):.2f} MB")

        self.save_results()
        
        
    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        for param in global_model.parameters():
            param.data.zero_()

        upload_cost = 0
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            client_model, size = load_item(client.role, 'global_model', client.save_folder_name, get_size=True)
            upload_cost += size
            for server_param, client_param in zip(global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * 1/len(self.uploaded_ids)

        print(f"Upload cost: {upload_cost:.2f} MB")
        self.upload_cost_list.append(upload_cost)

        save_item(global_model, self.role, 'global_model', self.save_folder_name)
