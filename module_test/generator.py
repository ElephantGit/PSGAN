import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from amm import AMM
from manet import DeMakeup, ReMakeup
from mdnet import MakeupDistillation 
from PIL import Image

class Generator_PS(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=6, input_inc=3):
        super(Generator_PS, self).__init__()
        self.amm = AMM()
        self.Demakeup = DeMakeup()
        self.Remakeup = ReMakeup()
        self.MDNet = MakeupDistillation()
    
    def forward(self, x, y, points_x, points_y, mask_x, mask_y):
        feat_x = self.Demakeup(x)
        feat_y = self.MDNet(y)
        feat_x_hat = self.amm(feat_x, feat_y, points_x, points_y, mask_x, mask_y)
        out = self.Remakeup(feat_x_hat)
        return out


if __name__ == '__main__':
    org = Image.open('111.png')

    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))

    points_x = torch.randn((1, 136, 64, 64))
    points_y = torch.randn((1, 136, 64, 64))

    mask_x = np.ones((1, 1, 64, 64))
    mask_y = np.zeros((1, 1, 64, 64))

    G = Generator_PS()
    x_hat = G(x, y, points_x, points_y, mask_x, mask_y)
    # print(x_hat)
    print(x_hat.shape)
