import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(dim_out, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(dim_out, affine=True)
                )
    def forward(self, x):
        return x + self.main(x)

class MakeupDistillation(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=6, input_nc=3):
        super(MakeupDistillation, self).__init__()

        layers_encoder = []
        layers_encoder.append(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers_encoder.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers_encoder.append(nn.ReLU(inplace=True))
        layers_encoder.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        layers_encoder.append(nn.InstanceNorm2d(conv_dim*2, affine=True))
        layers_encoder.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim*2
        layers_encoder.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        layers_encoder.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
        layers_encoder.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim * 2

        # Bottleneck-Encoder
        for i in range(repeat_num):
            layers_encoder.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.encoder = nn.Sequential(*layers_encoder)

    def forward(self, x):
        self.out = self.encoder(x)
        return self.out

if __name__ == '__main__':
    img = np.random.randn(1, 3, 256, 256)
    mdnet = MakeupDistillation()
    feat_ref = mdnet(torch.Tensor(img))
    print(feat_ref)
    print(feat_ref.shape)
