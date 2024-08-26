import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet32x32 import resnet18_32x32, resnet50_32x32
from torch.distributions import Bernoulli
from torch.hub import load_state_dict_from_url
from models.feature_extractor.vision_transformer import VisionTransformer
from functools import partial
import timm

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        # self.gamma = gamma
        # self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).to(x.device)
            # print((x.sample[-2], x.sample[-1]))
            block_mask = self._compute_block_mask(mask)
            # print (block_mask.size())
            # print (x.size())
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x


    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        # print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t().to(mask.device)
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).long().to(mask.device), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            # block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

def conv3x3(in_planes, out_planes, stride=(1, 1)):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        if avg_pool:
            #self.avgpool = nn.AvgPool2d(5, stride=1)
            self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def resnet12(keep_prob=1.0, avg_pool=False, drop_rate=0.0,**kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool,drop_rate=drop_rate, **kwargs)
    return model


class ServerModel(nn.Module):

    def __init__(self, encoder, client_class, total_class, pretrained):
        """
        Initialize the image model
        Args:
            encoder: str, encoder name
            client_class: int, number of classes
        """
        super(ServerModel, self).__init__()
        self.transform = None

        if encoder == "resnet50-cifar10" or encoder == "resnet50-cifar100" or encoder == "resnet50-smallkernel" or encoder == "resnet50":
            temp_model = resnet50_32x32()
            self.encoder = nn.Sequential(*list(temp_model.children())[:-1])
            num_feature = temp_model.fc.in_features
        elif encoder == "resnet18-cifar10" or encoder == "resnet18":
            print("Using resnet18")
            temp_model = resnet18_32x32()
            if pretrained:
                state_dict = load_state_dict_from_url(model_urls['resnet18'], progress=True)
                del state_dict['conv1.weight']
                logs = temp_model.load_state_dict(state_dict, strict=False)
                print(logs)
            self.encoder = nn.Sequential(*list(temp_model.children())[:-1])
            num_feature = temp_model.fc.in_features
        elif encoder == 'resnet12':
            self.encoder = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2)
            #num_feature=2560
            num_feature=640
        elif encoder == 'vit_base_patch16_clip_224.openai':
            self.transform = transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True)
            self.encoder = timm.create_model(
                encoder,
                pretrained=pretrained,
                img_size=224,
                num_classes=0)
            num_feature = 768
        elif encoder == 'clip_vit_tiny':
            self.encoder = VisionTransformer(
                img_size=32,
                patch_size=4,
                embed_dim=192,
                depth=12,
                num_heads=3,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            num_feature = 192
        else:
            raise "Unknown encoder"

        # trainable parameter for cosine similarity
        self.tau = nn.Parameter(torch.tensor(1.0))

        # projection MLP
        self.all_classify = nn.Linear(num_feature, total_class)


    def forward(self, x_input):
        """
        :param x_input: input image
        """
        if self.transform:
            x_input = self.transform(x_input)
        ebd = self.encoder(x_input)
        # remove all dimensions with size 1
        b, c = ebd.size(0), ebd.size(1)
        h = ebd.squeeze().view(b, c)
        y = self.all_classify(h)
        return h, y


class ClientModel(nn.Module):

    def __init__(self, encoder, num_classes):
        """
        Initialize the image model
        Args:
            encoder: str, encoder name
            num_classes: int, number of classes
        """
        super(ClientModel, self).__init__()

        if encoder == "ResNet18_32x32":
            resnet = resnet18_32x32()
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
            num_feature = resnet.fc.in_features
        elif 'clip' in encoder:
            self.encoder = VisionTransformer(
                img_size=32,
                patch_size=4,
                embed_dim=192,
                depth=12,
                num_heads=3,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            num_feature = 192
        else:
            raise "Unknown encoder"

        # trainable parameter for cosine similarity
        self.tau = nn.Parameter(torch.tensor(1.0))

        # projection MLP
        self.output = nn.Linear(num_feature, num_classes)

        # Docking layer
        self.docking = nn.Linear(num_feature, num_feature)


    def forward(self, x_input, use_docking=False):
        """
        :param x_input: input image
        :param use_docking: if True, use docking layer to project the feature, else use the original feature
        """
        ebd = self.encoder(x_input)
        # remove all dimensions with size 1
        b, c = ebd.size(0), ebd.size(1)
        h = ebd.squeeze().view(b, c)
        y = self.output(h)
        if use_docking:
            h = self.docking(h)
        return h, y
