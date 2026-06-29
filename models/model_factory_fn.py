import copy
import torch
import time

from models import generators
from models.model import HeteroModel


def get_decoder(model_name, **kwargs):
    if model_name == 'LatentGenerator':
        decoder = generators.LatentGenerator(**kwargs)
    elif model_name == 'ClassLatentGenerator':
        decoder = generators.ClassLatentGenerator(**kwargs)
    else:
        raise ValueError(f'Unknown decoder model: {model_name}.')
    size = get_model_size(decoder)
    parameter = count_parameters(decoder)
    print(f'Decoder {model_name}, size: {size:.3f}MB, # of parameters: {parameter}.')
    return decoder


def init_client_nets(num_clients, model_choice, n_class, docking_dim, get_pure_student=False, device='cpu'):
    """
    Initialize the networks for each client
    :param num_clients: Number of clients
    :param model_choice: List of model choices
    :param n_class: Number of classes in dataset
    :param get_pure_student: if True, return an extra student model, which by default is a copy of the first choice
    :param docking_dim: Dimension of the unified embedding space when translating using docking layer
    :param device: cpu or cuda
    :return: List of client models
    """
    n = len(model_choice)
    model_list = []
    for encoder in model_choice:
        model = HeteroModel(encoder, n_class, docking_dim)
        model.to(device)
        size = get_model_size(model)
        parameter = count_parameters(model)
        for param in model.parameters():
            param.requires_grad = True
        model_list.append(model)
        time_efficiency = estimate_model_efficiency(model)
        print(f'Model choice {encoder}, size: {size:.3f}MB, # of parameters: {parameter},'
              f' time efficiency: {time_efficiency:.3f}s')

    nets = []
    for net_i in range(num_clients):
        net = copy.deepcopy(model_list[net_i % n])
        net.to(device)
        nets.append(net)

    if get_pure_student:
        student = copy.deepcopy(model_list[0])
        student.to(device)
        nets.append(student)

    return nets


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb


def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.nelement()
    return total_params


def estimate_model_efficiency(model, image_size=128, device='cuda'):
    x = torch.randn(100, 3, image_size, image_size).to(device)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(x)
    return time.time() - start_time
