import logging
import math
import os

import numpy as np
import pandas as pd
import torch
import copy
from experiments.models.model import model_pull, generator_model_pull
from lightfed.core import BaseServer
from lightfed.tools.aggregator import ModelStateAvgAgg, NumericAvgAgg
from lightfed.tools.funcs import (consistent_hash, formula, save_pkl, model_size, set_seed, grad_False, grad_True)
from lightfed.tools.model import evaluation, get_buffers, get_parameters, get_cpu_param, DiversityLoss
from torch import nn, optim
from torch.autograd import Variable

from trainer import ClientTrainer
from collections import OrderedDict
import time
import torch.nn.functional as F



class ServerManager(BaseServer):
    def __init__(self, ct, args):
        super().__init__(ct)
        self.super_params = args.__dict__.copy()
        del self.super_params['data_distributor']
        del self.super_params['log_level']
        self.app_name = args.app_name
        self.device = args.device
        self.client_num = args.client_num                   
        self.selected_client_num = args.selected_client_num 
        self.comm_round = args.comm_round                  
        self.I = args.I                                     
        self.eval_step_interval = args.eval_step_interval
        self.temp = args.temp
        self.data_set = args.data_set
        self.model_heterogeneity = args.model_heterogeneity

        self.weight_agg_plus = args.weight_agg_plus

        self.model_split_mode = args.model_split_mode
        self.global_model_rate = args.global_model_rate     

        self.global_model_rate = args.global_model_rate     


        self.full_train_dataloader = args.data_distributor.get_train_dataloader()  
        self.full_test_dataloader = args.data_distributor.get_test_dataloader()    
        self.label_split = args.data_distributor.client_label_list                 

        self.local_sample_numbers = [len(args.data_distributor.get_client_train_dataloader(client_id).dataset)
                                     for client_id in range(args.client_num)]
        self.n_class = args.data_distributor.n_class

        set_seed(args.seed + 657)

        self.model = model_pull(self.super_params).to(self.device)  
        if self.weight_agg_plus == True:
            self.global_params = get_parameters(self.model.state_dict())
            if self.model_heterogeneity == True:
                self.tmp_counts = {}
                for k, v in self.global_params.items():
                    self.tmp_counts[k] = torch.ones_like(v)
            
            if self.model_heterogeneity == False:  
                self.global_params_aggregator = ModelStateAvgAgg()

        self.rate = args.rate

        self.selected_clients = None
        self.param_idx = None

        self.weighting = args.weighting

        self.model_idxs = {}
        self.roll_idx = {}

        torch.cuda.empty_cache()

        self.client_collect_list = []
        self.clients_bn_layers_collect = []

        self.client_test_acc_aggregator = NumericAvgAgg()

        self.comm_load = {client_id: 0 for client_id in range(args.client_num)}

        self.client_eval_info = []  
        self.global_train_eval_info = []  

        self.unfinished_client_num = -1


        self.generator = generator_model_pull(self.super_params).to(self.device)

        self.reload_generator = args.reload_generator

        self.batch_size = args.batch_size
        self.gen_lr = args.gen_lr
        self.gen_I = args.gen_I
        self.b1 = args.b1
        self.b2 = args.b2
        self.latent_dim = args.latent_dim

        self.global_lr = args.global_lr
        self.global_I = args.global_I

        self.adv_I = args.adv_I

        self.beta_bn = args.beta_bn
        self.beta_div = args.beta_div


        self.mom = args.mom

        self.label_num = []

        ##
        self.init_loss_fn()

        self.step = -1
    
    def init_loss_fn(self):
        self.KL_batchmean = nn.KLDivLoss(reduction="batchmean").to(self.device)
        self.CE = nn.CrossEntropyLoss().to(self.device)
        self.diversity_loss = DiversityLoss(metric='l2').to(self.device)
        self.KL = nn.KLDivLoss(reduce=False).to(self.device)  
        self.NLL = nn.NLLLoss(reduce=False).to(self.device)
        self.MSE = nn.MSELoss(reduce=False).to(self.device)

    def start(self):
        logging.info("start...")
        self.next_step()

    def end(self):
        logging.info("end...")

        self.super_params['device'] = self.super_params['device'].type

        ff = f"{self.app_name}-{consistent_hash(self.super_params, code_len=64)}.pkl"
        logging.info(f"output to {ff}")

        result = {'super_params': self.super_params,
                  'global_train_eval_info': pd.DataFrame(self.global_train_eval_info),
                  'client_eval_info': pd.DataFrame(self.client_eval_info),
                  'comm_load': self.comm_load}
        save_pkl(result, f"{os.path.dirname(__file__)}/Result/{ff}")

        self._ct_.shutdown_cluster()

    def end_condition(self):
        return self.step > self.comm_round - 1

    def next_step(self):
        self.step += 1
        self.selected_clients = self._new_train_workload_arrage()  
        self.unfinished_client_num = self.selected_client_num

        if self.weight_agg_plus == False:
            for id_ in range(len(self.selected_clients)):
                client_id = self.selected_clients[id_]
                self._ct_.get_node('client', client_id) \
                    .fed_client_train_step(step=self.step, global_params=None) 
        else:
            if self.model_heterogeneity == True:
                local_parameters, self.param_idx = self.distribute() 
                for id_ in range(len(self.selected_clients)):
                    client_id = self.selected_clients[id_]
                    self._ct_.get_node('client', client_id) \
                        .fed_client_train_step(step=self.step, global_params=local_parameters[id_]) 
                torch.cuda.empty_cache()
            else:
                for id_ in range(len(self.selected_clients)):
                    client_id = self.selected_clients[id_]
                    self._ct_.get_node('client', client_id) \
                        .fed_client_train_step(step=self.step, global_params=self.global_params) 

    def _new_train_workload_arrage(self):
        if self.selected_client_num < self.client_num:
            selected_client = np.random.choice(range(self.client_num), self.selected_client_num, replace=False)
        elif self.selected_client_num == self.client_num:
            selected_client = np.array([i for i in range(self.client_num)])
        return selected_client
    

    def client_bn_layer_mean_var(self, client_model):
        client_bn_layers = []
        for module in client_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                client_bn_layers.append((module.running_mean.clone().detach(), module.running_var.clone().detach()))
        return client_bn_layers

    def fed_finish_client_train_step(self,
                                     step,
                                     client_id,
                                     client_bn_layers,
                                     client_model,
                                     eval_info):
        logging.debug(f"train comm. round of client_id:{client_id} comm. round:{step} was finished")
        assert self.step == step
        self.client_eval_info.append(eval_info)

        weight = self.local_sample_numbers[client_id]

        if self.data_set in ['FOOD101', 'Tiny-Imagenet']:
            if self.step % 5 == 0:
                self.client_test_acc_aggregator.put(eval_info['test_acc'], weight)
        else:
            self.client_test_acc_aggregator.put(eval_info['test_acc'], weight)

        if self.weight_agg_plus == True:
            if self.model_heterogeneity == False:
                client_model_params = get_parameters(client_model.state_dict())
                self.global_params_aggregator.put(client_model_params, weight)

        grad_False(client_model)  
        self.client_collect_list.append((client_id, client_model))

        self.clients_bn_layers_collect.append((client_id, client_bn_layers))


        if self.comm_load[client_id] == 0:
            client_model_params = get_cpu_param(client_model.state_dict())
            self.comm_load[client_id] = model_size(client_model_params) / 1024 / 1024  

        self.unfinished_client_num -= 1
        if not self.unfinished_client_num:
            self.server_train_test_res = {'comm. round': self.step, 'client_id': 'server'}

            if self.weight_agg_plus == True:
                if self.model_heterogeneity == True:
                    self.combine(self.client_collect_list, self.param_idx)
                else:
                    self.global_params = self.global_params_aggregator.get_and_clear()
                self.model.load_state_dict(self.global_params, strict=True)
                torch.cuda.empty_cache()


            self.FloatTensor = torch.cuda.FloatTensor
            self.LongTensor = torch.cuda.LongTensor
            for _ in range(self.adv_I):

                self.Update_generator()
                self.Update_global_model()

            if self.I == 0:
                if self.step % self.eval_step_interval == 0:
                    client_test_acc_avg = self.client_test_acc_aggregator.get_and_clear()
                    print('comm. round: {}, client_test_acc: {}'.format(self.step, client_test_acc_avg))
                    self._set_global_train_eval_info()
                    self.global_train_eval_info.append(self.server_train_test_res)
            else:
                if self.data_set in ['FOOD101', 'Tiny-Imagenet']:
                    if self.step % 5 == 0:
                        client_test_acc_avg = self.client_test_acc_aggregator.get_and_clear()
                        print('comm. round: {}, client_test_acc: {}'.format(self.step, client_test_acc_avg))
                        self._set_global_train_eval_info()
                        self.global_train_eval_info.append(self.server_train_test_res)
                else:
                    client_test_acc_avg = self.client_test_acc_aggregator.get_and_clear()
                    print('comm. round: {}, client_test_acc: {}'.format(self.step, client_test_acc_avg))
                    self._set_global_train_eval_info()
                    self.global_train_eval_info.append(self.server_train_test_res)

            logging.debug(f"train comm. round:{step} is finished")
            
            self.server_train_test_res = {}

            self.clients_bn_layers_collect = []
            self.client_collect_list = []
            self.next_step()

    def Update_generator(self, ):
        if self.reload_generator == True:
            self.generator = generator_model_pull(self.super_params).to(self.device)
        grad_True(self.generator)
        self.generator.train()
        grad_False(self.model)
        self.model.eval()
        LOSS_GEN = 0
        L_CE, L_BN, L_DIV = 0, 0, 0 
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.gen_lr, betas=(self.b1, self.b2))
        for _ in range(self.gen_I):
            self.generator.zero_grad(set_to_none=True)
            self.optimizer_G.zero_grad(set_to_none=True)

            z = Variable(self.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))
            y = np.random.choice(self.n_class, self.batch_size)
            gen_labels = Variable(self.LongTensor(np.array(y)))

            gen_imgs, _, _ = self.generator(z, gen_labels)


            global_logits = self.model(gen_imgs)
            global_logits_softmax = F.softmax(global_logits / self.temp, dim=1).clone().detach()

            _, global_pre = torch.max(global_logits, -1)
            global_pre_res = global_pre.eq(gen_labels)

            L_bn = 0
            L_kl_logit = 0
            for client_bn, client_model in zip(self.clients_bn_layers_collect, self.client_collect_list):
                client_idx = client_model[0]
                local_model = client_model[1]
                local_model.eval()
                client_logits, bn_input_list = local_model(gen_imgs, label_list=self.label_split[client_idx], bn_or_not=True)
                
                gen_client_bn_layers = []
                for bn_i in bn_input_list:
                    mean = bn_i.mean([0, 2, 3]) 
                    var = bn_i.var([0, 2, 3]) 
                    gen_client_bn_layers.append((mean, var))
                len_ = len(gen_client_bn_layers)
                local_bn_loss = 0
                for gen_mean_var, mean_var in zip(gen_client_bn_layers, client_bn[1]):
                    local_bn_loss += torch.norm(gen_mean_var[0] - mean_var[0], 2) + torch.norm(gen_mean_var[1] - mean_var[1], 2)
                L_bn += local_bn_loss / self.selected_client_num / len_

                L_kl_logit += client_logits / self.selected_client_num

            L_ce = self.CE(L_kl_logit, gen_labels)

            _, ensemble_pre = torch.max(L_kl_logit, -1)
            ensemble_pre_res = ensemble_pre.eq(gen_labels)

            varepsilon = []
            for idx, pre in enumerate(zip(ensemble_pre_res, global_pre_res)):
                en_pre, gl_pre = pre[0], pre[1]
                if (en_pre == True and gl_pre == False) or (en_pre == False and gl_pre == True):
                    varepsilon.append(idx)
                else:
                    continue
                    

            clients_logit_log_softmax = F.log_softmax(L_kl_logit / self.temp, dim=1)
            L_div = - self.KL_batchmean(clients_logit_log_softmax[varepsilon], global_logits_softmax[varepsilon]) 
            loss_gen =  L_ce + self.beta_bn * L_bn + self.beta_div * L_div 
            loss_gen.backward()
            self.optimizer_G.step()

            LOSS_GEN += loss_gen.detach().cpu().numpy()
            L_CE += L_ce.detach().cpu().numpy()
            L_BN += L_bn.detach().cpu().numpy()
            L_DIV += L_div.detach().cpu().numpy()
        info_gen = {'comm. round': self.step, 'client_id': 'server', 'update': 'generator',
                    'LOSS_GEN': round(LOSS_GEN / self.gen_I, 4), 'L_CE': round(L_CE / self.gen_I, 4), 'L_BN': round(L_BN / self.gen_I, 4),
                    'L_DIV': round(L_DIV / self.gen_I, 4)} 
        self.server_train_test_res.update(generator_update=info_gen)
    
    def Update_global_model(self,):
        grad_False(self.generator)
        self.generator.eval()

        grad_True(self.model)
        self.model.train()
        L_DIS = 0.0
        self.optimizer_model = optim.SGD(params=self.model.parameters(), lr=self.global_lr, weight_decay=0.0001)
        for _ in range(self.global_I):
            self.model.zero_grad(set_to_none=True)
            self.optimizer_model.zero_grad(set_to_none=True)

            z = Variable(self.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))
            y = np.random.choice(self.n_class, self.batch_size)
            gen_labels = Variable(self.LongTensor(np.array(y)))

            L_dis = self.cal_L_kl(self.generator, z, y, gen_labels)

            L_dis.backward()
            self.optimizer_model.step()
            L_DIS += L_dis.detach().cpu().numpy()

        info_global = {'comm. round': self.step, 'client_id': 'server', 'update': 'global_model', 'L_DIS': round(L_DIS / self.gen_I, 4)}

        self.server_train_test_res.update(global_model_update=info_global)

    
    def cal_L_kl(self, generator, z, y, gen_labels):
        gen_imgs, _, _ = generator(z, gen_labels)
        global_logits = self.model(gen_imgs)
        global_logits_log_softmax = F.log_softmax(global_logits / self.temp, dim=1)

        L_kl_logit = 0
        for client_idx, local_model in self.client_collect_list:
            local_model.eval()
            client_logits = local_model(gen_imgs, label_list=self.label_split[client_idx])
            
            L_kl_logit += client_logits / self.selected_client_num

        clients_logit_softmax = F.softmax(L_kl_logit / self.temp, dim=1).clone().detach()
        L_kl = self.KL_batchmean(global_logits_log_softmax, clients_logit_softmax) 
        return L_kl
    
    def distribute(self,):
        self.model_rate = np.array(self.rate)
        if self.model_split_mode == 'roll':
            param_idx = self.split_model_roll()
        elif self.model_split_mode == 'static':
            param_idx = self.split_model_static()
        elif self.model_split_mode == 'random':
            param_idx = self.split_model_random()

        self.global_params = get_parameters(self.model.state_dict())
        local_parameters = [OrderedDict() for _ in range(len(self.selected_clients))]
        for k, v in self.global_params.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(self.selected_clients)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            local_parameters[m][k] = copy.deepcopy(v[torch.meshgrid(param_idx[m][k])])
                        else:
                            local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                    else:
                        local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                elif parameter_type in ['running_mean', 'running_var']:
                    local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                else:
                    local_parameters[m][k] = copy.deepcopy(v)
        return local_parameters, param_idx
     
    def split_model_roll(self):
        idx_i = [None for _ in range(len(self.selected_clients))]
        idx = [OrderedDict() for _ in range(len(self.selected_clients))]
        for k, v in self.global_params.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(self.selected_clients)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'conv' in k: 
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[self.selected_clients[m]] / self.global_model_rate
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                
                                roll = self.step % output_size
                                model_idx = torch.arange(output_size, device=v.device)
                                
                                
                                model_idx = torch.roll(model_idx, roll, -1)
                                output_idx_i_m = model_idx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            elif 'shortcut' in k:
                                input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                output_idx_i_m = idx_i[m]
                            elif 'linear' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                raise ValueError('Not valid k')
                            idx[m][k] = (output_idx_i_m, input_idx_i_m) 
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'linear' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                elif parameter_type in ['running_mean', 'running_var']:
                    input_idx_i_m = idx_i[m]
                    idx[m][k] = input_idx_i_m

                else:
                    pass
        return idx
    
    def split_model_static(self):
        idx_i = [None for _ in range(len(self.selected_clients))]
        idx = [OrderedDict() for _ in range(len(self.selected_clients))]
        for k, v in self.global_params.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(self.selected_clients)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'conv' in k:
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[self.selected_clients[m]] / self.global_model_rate
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                model_idx = torch.arange(output_size, device=v.device)
                                output_idx_i_m = model_idx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            elif 'shortcut' in k:
                                input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                output_idx_i_m = idx_i[m]
                            elif 'linear' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                raise ValueError('Not valid k')
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'linear' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                elif parameter_type in ['running_mean', 'running_var']:
                    input_idx_i_m = idx_i[m]
                    idx[m][k] = input_idx_i_m

                else:
                    pass
        return idx
    
    def split_model_random(self):
        idx_i = [None for _ in range(len(self.selected_clients))]
        idx = [OrderedDict() for _ in range(len(self.selected_clients))]
        for k, v in self.global_params.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(self.selected_clients)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'conv' in k: 
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[self.selected_clients[m]] / self.global_model_rate
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                model_idx = torch.randperm(output_size, device=v.device)
                                output_idx_i_m = model_idx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            elif 'shortcut' in k:
                                input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                output_idx_i_m = idx_i[m]
                            elif 'linear' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                raise ValueError('Not valid k')
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'linear' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m

                elif parameter_type in ['running_mean', 'running_var']:
                    input_idx_i_m = idx_i[m]
                    idx[m][k] = input_idx_i_m

                else:
                    pass
        return idx

    def combine(self, local_models, param_idx): 
        count = OrderedDict()
        updated_parameters = copy.deepcopy(self.global_params) 
        tmp_counts_cpy = copy.deepcopy(self.tmp_counts)        
        for k, v in updated_parameters.items():
            parameter_type = k.split('.')[-1]
            count[k] = v.new_zeros(v.size(), dtype=torch.float32, device=self.device)
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32, device=self.device)
            for m, local_m in enumerate(local_models):
                local_pars = get_parameters(local_m[1].state_dict())
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            if 'linear' in k:
                                label_split = self.label_split[local_m[0]]  
                                param_idx[m][k] = list(param_idx[m][k])
                                param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                tmp_v[torch.meshgrid(param_idx[m][k])] += self.tmp_counts[k][torch.meshgrid(param_idx[m][k])] * local_pars[k][label_split]
                                count[k][torch.meshgrid(param_idx[m][k])] += self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                tmp_counts_cpy[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                output_size = v.size(0)
                                scaler_rate = self.model_rate[local_m[0]] / self.global_model_rate
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                if self.weighting == 'avg':
                                    K = 1
                                elif self.weighting == 'width':
                                    K = local_output_size
                                elif self.weighting == 'updates':
                                    K = self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                elif self.weighting == 'updates_width':
                                    K = local_output_size * self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]

                                tmp_v[torch.meshgrid(param_idx[m][k])] += K * local_pars[k]
                                count[k][torch.meshgrid(param_idx[m][k])] += K
                                tmp_counts_cpy[k][torch.meshgrid(param_idx[m][k])] += 1
                        else:
                            tmp_v[param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]] * local_pars[k]
                            count[k][param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]]
                            tmp_counts_cpy[k][param_idx[m][k]] += 1
                    else:
                        if 'linear' in k:
                            label_split = self.label_split[local_m[0]]
                            param_idx[m][k] = param_idx[m][k][label_split]
                            tmp_v[param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]] * local_pars[k][label_split]
                            count[k][param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]]
                            tmp_counts_cpy[k][param_idx[m][k]] += 1
                        else:
                            tmp_v[param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]] * local_pars[k]
                            count[k][param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]]
                            tmp_counts_cpy[k][param_idx[m][k]] += 1
                elif parameter_type in ['running_mean', 'running_var']:
                    tmp_v[param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]] * local_pars[k]
                    count[k][param_idx[m][k]] += self.tmp_counts[k][param_idx[m][k]]
                    tmp_counts_cpy[k][param_idx[m][k]] += 1
                else:
                    tmp_v += self.tmp_counts[k] * local_pars[k]
                    count[k] += self.tmp_counts[k]
                    tmp_counts_cpy[k] += 1
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
            self.tmp_counts = copy.deepcopy(tmp_counts_cpy)
        self.global_params = updated_parameters
        return
        
    def _set_global_train_eval_info(self):
        loss, acc, num = evaluation(model=self.model,
                                    dataloader=self.full_test_dataloader,
                                    criterion=self.CE,
                                    device=self.device,
                                    eval_full_data=True)
            
        torch.cuda.empty_cache()
        self.server_train_test_res.update(test_loss=loss, test_acc=acc, test_sample_size=num)
        test_ = {}
        test_.update(test_loss=loss, test_acc=acc, test_sample_size=num)
        print(test_)


class ClientManager(BaseServer):
    def __init__(self, ct, args):
        super().__init__(ct)
        self.I = args.I
        self.device = args.device
        self.client_id = self._ct_.role_index
        self.model_type = args.model_type
        self.data_set = args.data_set
        self.weight_agg_plus = args.weight_agg_plus

        self.args = args.__dict__.copy()
        del self.args['data_distributor']

        self.trainer = ClientTrainer(args, self.client_id)
        self.step = 0

    def start(self):
        logging.info("start...")

    def end(self):
        logging.info("end...")

    def end_condition(self):
        return False

    def fed_client_train_step(self, step, global_params):
        self.step = step
        logging.debug(f"training client_id:{self.client_id}, comm. round:{step}")

        self.trainer.res = {'communication round': step, 'client_id': self.client_id}
        
        if self.weight_agg_plus == False:
            if step == 0:
                self.trainer.pull_local_model(self.args, model_rate = self.args['rate'][self.client_id])  
        else:
            self.trainer.pull_local_model(self.args, model_rate = self.args['rate'][self.client_id])  
            self.trainer.model.load_state_dict(global_params, strict=True)

        self.timestamp = time.time()
        self.trainer.train_locally_step(self.I, step)
        curr_timestamp = time.time()
        train_time = curr_timestamp - self.timestamp

        self.finish_train_step(self.trainer.model, self.trainer.gen_client_bn_layers, train_time)
        self.trainer.clear()
        torch.cuda.empty_cache()

    def finish_train_step(self, model, client_bn_layers, train_time):
        self.trainer.get_eval_info(self.step, train_time)
        logging.debug(f"finish_train_step comm. round:{self.step}, client_id:{self.client_id}")

        self._ct_.get_node("server") \
            .set(deepcopy=False) \
            .fed_finish_client_train_step(self.step,
                                          self.client_id,
                                          client_bn_layers,
                                          model,
                                          self.trainer.res)