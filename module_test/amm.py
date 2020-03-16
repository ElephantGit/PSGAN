import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

class AMM(nn.Module):
    def __init__(self, visual_weight=0.01):
        super(AMM, self).__init__()
        self.conv1 = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(256, 1, kernel_size=1, stride=1)

        self.visual_weight = visual_weight

    def forward(self, feat_src, feat_ref, landmarks_src, landmarks_ref, mask_src, mask_ref):
        print(feat_ref.shape)
        print(feat_src.shape)
        print(landmarks_ref.shape)
        print(landmarks_src.shape)
        print(mask_ref.shape)
        print(mask_src.shape)
        self.feat_src = feat_src
        self.feat_ref = feat_ref
        self.points_src = landmarks_src
        self.points_ref = landmarks_ref
        self.mask_src = mask_src
        self.mask_ref = mask_ref

        beta = self.conv1(self.feat_ref)
        print('beta shape:', beta.shape)
        gama = self.conv2(self.feat_ref)
        print('gama shape:', gama.shape)

        b, c, h, w = self.feat_src.size()
        print(b, c, h, w)

        # feat_src: (h,w,c) (h,w,136) --> (h*w, c+136)
        # feat_ref: (h,w,c) (h,w,136) --> (c+136, h*w)
        src = torch.cat(tensors=(self.visual_weight * self.feat_src, self.points_src), dim=1)
        feat_points_src = torch.reshape(src, (h * w, c + 136))
        print('shape of feat_points_src: ',feat_points_src.shape)
        ref = torch.cat(tensors=(self.visual_weight * self.feat_ref, self.points_ref), dim=1)
        feat_points_ref = torch.reshape(ref, (c + 136, h * w))
        print('shape of feat_points_ref: ', feat_points_ref.shape)
        
        self.mask_src = np.reshape(self.mask_src, (h * w))
        self.mask_ref = np.reshape(self.mask_ref, (h * w))
        M = []
        for i, m in enumerate(self.mask_src):
            M.append(self.mask_ref[i] == m)
        M = torch.Tensor(np.array(M).astype(np.float32))
        print('shape of M:', M.shape)

        # calculate Attention Map
        # A = (h*w, h*w) = (h*w, c+136) * (c+136, h*w)
        A = torch.mm(feat_points_src, feat_points_ref)
        A = F.softmax(A, dim=0)
        print('shape of A: ', A.shape)
        print('A before multi M: ', A)

        # apply mask
        A = A * M
        print('A after multi M: ', A)

        # makeup morphing(beta-->beta', gama-->gama')
        # (1,h,w) --> (h*w, h*w) * (h*w, 1) --> (h*w, 1) --> (1, h, w)
        beta_hat = torch.mm(A, torch.reshape(beta, (h * w, 1)))
        beta_hat = torch.reshape(beta_hat, (1, h, w))
        gama_hat = torch.mm(A, torch.reshape(gama, (h * w, 1)))
        gama_hat = torch.reshape(gama_hat, (1, h, w))

        # makeup transfer process
        c, h, w = gama_hat.shape
        feat_src_trans = torch.add(gama_hat.expand((1, c, h, w)) * feat_src, beta_hat)

        return feat_src_trans

if __name__ == '__main__':
    img = Image.open('111.png').convert('RGB')
    mask = Image.open('111_mask.png').convert('RGB')
    img = img.resize((256, 256))
    mask = mask.resize((256, 256))

    points_src = torch.ones((1, 136, 64, 64))
    points_ref = torch.ones((1, 136, 64, 64))

    feat_src = torch.rand((1, 256, 64, 64))
    feat_ref = torch.rand((1, 256, 64, 64))

    mask_src = np.ones((1, 1, 64, 64))
    mask_ref = np.zeros((1, 1, 64, 64))

    amm = AMM()

    feat_src_trans = amm.forward(feat_src, feat_ref, points_src, points_ref, mask_src, mask_ref)
    print(feat_src_trans.shape)


