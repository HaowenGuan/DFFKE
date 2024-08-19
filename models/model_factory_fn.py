from models import resnet
# from models import simple_models
# from models import vgg
from models import generators
from models.model import *
import copy
# import torchsummary


def get_model(model_name, **kwargs):
    if model_name == 'resnet18':
        return resnet.resnet18(**kwargs)
    else:
        raise ValueError('Wrong model name.')


def get_generator(model_name, **kwargs):
    if model_name == 'CGeneratorA':
        return generators.CGeneratorA(**kwargs)
    elif model_name == 'LatentGenerator':
        return generators.LatentGenerator(**kwargs)
    elif model_name == 'ClassLatentGenerator':
        return generators.CLassLatentGenerator(**kwargs)
    else:
        print(model_name)
        raise ValueError('Wrong model name.')


def init_client_nets(num_clients, client_encoder, n_class, device):
    """
    Initialize the networks for each client
    """
    nets = {net_i: None for net_i in range(num_clients)}

    model = ClientModel(client_encoder, n_class)
    model.to(device)
    size = get_model_size(model)
    print(f'Client model {client_encoder} size: {size:.3f}MB')
    for param in model.parameters():
        param.requires_grad = True
    # torchsummary.summary(model, input_size=(3, 32, 32))

    for net_i in range(num_clients):
        net = copy.deepcopy(model)
        net.to(device)
        nets[net_i] = net

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