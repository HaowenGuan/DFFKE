from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def get_cifar10_transform(resize_img=128):
    train_transform = transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.Resize(resize_img),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                             std=[0.24703223, 0.24348513, 0.26158784])
    ])
    test_transform = transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize(resize_img),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                             std=[0.24703223, 0.24348513, 0.26158784])
    ])
    return {
        'train_transform': train_transform,
        'test_transform': test_transform,
    }


def get_cifar100_transform(resize_img=128):
    train_transform = transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.Resize(resize_img),
        transforms.RandomHorizontalFlip(),
        lambda x: np.array(x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    ])
    test_transform = transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize(resize_img),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    ])
    return {
        'train_transform': train_transform,
        'test_transform': test_transform,
    }


def get_tiny_imagenet_transform(resize_img=128):
    train_transform = transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(64, padding=8),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.Resize(resize_img),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize(resize_img),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        # transforms.Normalize(mean=[0.48023694, 0.44806704, 0.3975036],
        #                      std=[0.27643643, 0.26886328, 0.28158993])
    ])
    return {
        'train_transform': train_transform,
        'test_transform': test_transform,
    }
