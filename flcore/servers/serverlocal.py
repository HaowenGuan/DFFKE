import time
from flcore.clients.clientlocal import clientLocal
from flcore.servers.serverbase import Server
from threading import Thread


class Local(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientLocal)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.n_clients}")
        print("Finished creating server and clients.")
        
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            print(f"\n-------------Round number: {i}-------------")

            for client in self.selected_clients:
                client.train()

            if i % self.eval_gap == 0:
                print("\nEvaluate heterogeneous models after local training")
                self.evaluate()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], auto_break_patient=self.auto_break_patient):
                break


        print("\nBest accuracy.")
        print(f'{max(self.rs_test_acc):.2f}')
        print("Average time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
