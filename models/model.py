import torch
from torch import nn

################################ Following models will be used through eval() function ################################
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from torchvision.models import swin_t, efficientnet_v2_s, efficientnet_v2_m, googlenet, mobilenet_v2
from torchvision.models.vision_transformer import VisionTransformer
from models.resnet32x32 import resnet18_32x32, resnet34_32x32, resnet50_32x32, resnet101_32x32, resnet152_32x32
from models.resnet import resnet18, resnet34, resnet50


class HeteroModel(nn.Module):
    def __init__(self, encoder, num_classes, docking_dim=512):
        """
        Initialize the image model
        Args:
            encoder: str, encoder name
            num_classes: int, number of classes
        """
        super(HeteroModel, self).__init__()
        self.docking_dim = docking_dim
        self.encoder_name = encoder

        net = eval(encoder)

        if 'resnet' in encoder:
            self.encoder = nn.Sequential(*list(net.children())[:-1], nn.Flatten(1))
            num_feature = net.fc.in_features
        elif 'efficientnet' in encoder:
            self.encoder = nn.Sequential(*list(net.children())[:-1], nn.Flatten(1))
            num_feature = net.classifier[1].in_features
        elif 'mobilenet_v3' in encoder:
            self.encoder = nn.Sequential(*list(net.children())[:-1], nn.Flatten(1))
            num_feature = net.classifier[0].in_features
        elif 'mobilenet_v2' in encoder:
            self.encoder = nn.Sequential(*list(net.children())[:-1], nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
            num_feature = net.classifier[1].in_features
        elif 'shufflenet_v2' in encoder:
            self.encoder = nn.Sequential(*list(net.children())[:-1], nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
            num_feature = net.fc.in_features
        elif 'vit' in encoder:
            num_feature = net.hidden_dim
            self.encoder = nn.Sequential(net)
            # num_feature = docking_dim # , nn.Linear(num_feature, docking_dim), nn.Tanh()
        elif 'googlenet' in encoder:
            num_feature = net.fc.in_features
            net.fc = nn.Identity()
            self.encoder = nn.Sequential(net)
        elif 'FedGH_CNN' in encoder:
            num_feature = net.fc2.out_features
            self.encoder = nn.Sequential(net)
        elif 'FedAvgCNN' in encoder:
            num_feature = net.fc.in_features
            self.encoder = nn.Sequential(net)
        else:
            raise f'Unknown encoder: {encoder}'

        # Clamp the embedding vector to [0, 1]
        self.activation = nn.Hardtanh(0, 1)

        # Classification layer
        self.output = nn.Linear(num_feature, num_classes)

        # Docking layer
        self.docking = nn.Linear(num_feature, docking_dim)

    def forward(self, x_input, use_docking=False):
        """
        :param x_input: input image
        :param use_docking: if True, use docking layer to project the feature, else use the original feature
        """
        ebd = self.encoder(x_input)
        # ebd = self.clamp_vector_norm(ebd, self.docking_dim)
        # clamp_ebd = self.activation(ebd)
        y = self.output(ebd)
        if use_docking:
            ebd = self.docking(ebd)
        return ebd, y


class BaseHeadSplit(nn.Module):
    # split an original model into a base and a head
    def __init__(self, args, cid):
        super().__init__()

        base_name = args.models[cid % len(args.models)]
        base_model = eval(base_name)

        if 'resnet' in base_name:
            # encoder = nn.Sequential(*list(base_model.children())[:-1], nn.Flatten(1))
            # num_feature = base_model.fc.in_features
            base_model.fc = nn.AdaptiveAvgPool1d(args.feature_dim)
        elif 'efficientnet' in base_name:
            # encoder = nn.Sequential(*list(base_model.children())[:-1], nn.Flatten(1))
            # num_feature = base_model.classifier[1].in_features
            base_model.classifier = nn.AdaptiveAvgPool1d(args.feature_dim)
        elif 'mobilenet_v3' in base_name:
            # encoder = nn.Sequential(*list(base_model.children())[:-1], nn.Flatten(1))
            # num_feature = base_model.classifier[0].in_features
            base_model.classifier = nn.AdaptiveAvgPool1d(args.feature_dim)
        elif 'shufflenet_v2' in base_name:
            # encoder = nn.Sequential(*list(base_model.children())[:-1], nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
            # num_feature = base_model.fc.in_features
            base_model.fc = nn.AdaptiveAvgPool1d(args.feature_dim)
        elif 'vit' in base_name:
            # num_feature = base_model.hidden_dim
            # encoder = nn.Sequential(base_model)
            base_model.heads = nn.AdaptiveAvgPool1d(args.feature_dim)
        elif 'googlenet' in base_name:
            # num_feature = base_model.fc.in_features
            # base_model.fc = nn.Identity()
            # encoder = nn.Sequential(base_model)
            base_model.fc = nn.AdaptiveAvgPool1d(args.feature_dim)
        else:
            raise f'Unknown base model: {base_name}'

        # self.base = nn.Sequential(encoder, nn.Linear(num_feature, args.feature_dim), nn.Tanh())
        self.base = base_model

        self.head = nn.Linear(args.feature_dim, args.num_classes)

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out


def vit_tiny_torch(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        num_layers=4,
        num_heads=16,
        hidden_dim=192,
        mlp_dim=768,
        **kwargs)
    model.heads = nn.Identity()
    return model


class FedGH_CNN(nn.Module):  # for homo. exp.
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(FedGH_CNN, self).__init__()
        down_scale = 128 // 32
        self.downsample = nn.AvgPool2d(down_scale, stride=down_scale)
        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, out_dim)  # Original cls head, unused
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.downsample(x)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)  # instead of x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        down_scale = 128 // 32
        self.downsample = nn.AvgPool2d(down_scale, stride=down_scale)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)  # Original cls head, unused

    def forward(self, x):
        x = self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        # out = self.fc(out)
        return out
