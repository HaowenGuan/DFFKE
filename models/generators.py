import torch
import torch.nn as nn
import torch.nn.functional as F

class CGeneratorA(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32, n_cls=100):
        super(CGeneratorA, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*self.init_size**2))
        self.l2 = nn.Sequential(nn.Linear(n_cls, ngf*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False)
        )

    def forward(self, z, y):
        out_1 = self.l1(z.view(z.shape[0],-1)) # (batch_size, ngf*init_size**2)
        out_2 = self.l2(y.view(y.shape[0],-1)) # (batch_size, ngf*init_size**2)
        out = torch.cat([out_1, out_2], dim=1) # (batch_size, 2*ngf*init_size**2)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size) # (batch_size, 2*ngf, init_size, init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img



class LatentGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3, n_cls=100):
        super(LatentGenerator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, 2*ngf*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf*2, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, y):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class CLassLatentGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3, n_cls=100):
        super(CLassLatentGenerator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*self.init_size**2))
        self.l2 = nn.Sequential(nn.Linear(n_cls, ngf*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf*2, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, y):
        out_1 = self.l1(z.view(z.shape[0],-1)) # (batch_size, ngf*init_size**2)
        out_2 = self.l2(y.view(y.shape[0],-1)) # (batch_size, ngf*init_size**2)
        out = torch.cat([out_1, out_2], dim=1) # (batch_size, 2*ngf*init_size**2)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size) # (batch_size, 2*ngf, init_size, init_size)
        img = self.conv_blocks(out)
        return img