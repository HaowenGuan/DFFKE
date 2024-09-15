import random
import timm
import torch
import torch.nn.functional as F

from PIL import Image, ImageFilter
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def get_cifar10_transform():
    train_transform = transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                             std=[0.24703223, 0.24348513, 0.26158784])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                             std=[0.24703223, 0.24348513, 0.26158784])
    ])
    return {
        'train_transform': train_transform,
        'test_transform': test_transform,
    }


def get_cifar100_transform():
    train_transform = transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.array(x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    ])
    return {
        'train_transform': train_transform,
        'test_transform': test_transform,
    }


def get_vit_224_size_transform(timm_model):
    # data_config = timm.data.resolve_model_data_config(timm_model)
    # transform  = timm.data.create_transform(**data_config, is_training=False)
    # train_transform = test_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     *transform.transforms[2:]
    # ])
    train_transform = test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758])
    ])
    return {
        'train_transform': train_transform,
        'test_transform': test_transform,
    }


def get_vit_original_size_transform():
    train_transform = test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    ])
    return {
        'train_transform': train_transform,
        'test_transform': test_transform,
    }
