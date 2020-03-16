import os

import argparse
import time

from PIL import Image
import torch
from torch.backends import cudnn
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms

import net

from config_test import config, dataset_config, merge_cfg_arg
from dataloder import get_loader
from solver_cycle import Solver_cycleGAN
from solver_makeup import Solver_makeupGAN
from solver_psgan import Solver_PSGAN

def parse_args():
    parser = argparse.ArgumentParser(description='Train GAN')
    # general
    parser.add_argument('--data_path', default='makeup/test_data/', type=str, help='training and test data path')
    parser.add_argument('--dataset', default='MAKEUP', type=str, help='dataset name, MAKEUP means two domain, MMAKEUP means multi-domain')
    parser.add_argument('--gpus', default='0', type=str, help='GPU device to train with')
    parser.add_argument('--batch_size', default='1', type=int, help='batch_size')
    parser.add_argument('--vis_step', default='1260', type=int, help='steps between visualization')
    parser.add_argument('--task_name', default='', type=str, help='task name')
    parser.add_argument('--ndis', default='1', type=int, help='train discriminator steps')
    parser.add_argument('--LR', default="2e-4", type=float, help='Learning rate')
    parser.add_argument('--decay', default='0', type=int, help='epochs number for training')
    parser.add_argument('--model', default='makeupGAN', type=str, help='which model to use: cycleGAN/ makeupGAN')
    parser.add_argument('--epochs', default='300', type=int, help='nums of epochs')
    parser.add_argument('--whichG', default='branch', type=str, help='which Generator to choose, normal/branch, branch means two input branches')
    parser.add_argument('--norm', default='SN', type=str, help='normalization of discriminator, SN means spectrum normalization, none means no normalization')
    parser.add_argument('--d_repeat', default='3', type=int, help='the repeat Res-block in discriminator')
    parser.add_argument('--g_repeat', default='6', type=int, help='the repeat Res-block in Generator')
    parser.add_argument('--lambda_cls', default='1', type=float, help='the lambda_cls weight')
    parser.add_argument('--lambda_rec', default='10', type=int, help='lambda_A and lambda_B')
    parser.add_argument('--lambda_his', default='1', type=float, help='histogram loss on lips')
    parser.add_argument('--lambda_skin_1', default='0.1', type=float, help='histogram loss on skin equals to lambda_his* lambda_skin')
    parser.add_argument('--lambda_skin_2', default='0.1', type=float, help='histogram loss on skin equals to lambda_his* lambda_skin')
    parser.add_argument('--lambda_eye', default='1', type=float, help='histogram loss on eyes equals to lambda_his*lambda_eye')
    parser.add_argument('--content_layer', default='r41', type=str, help='vgg layer we use')
    parser.add_argument('--lambda_vgg', default='5e-3', type=float, help='the param of vgg loss')
    parser.add_argument('--cls_list', default='A_OM,B_OM', type=str, help='the classes we choose')
    parser.add_argument('--direct', action="store_true", default=False, help='direct means to add local cosmetic loss at the first, unified training')
    parser.add_argument('--finetune', action="store_true", default=False, help='finetune the network or not')
    parser.add_argument('--lips', action="store_true", default=False, help='whether to finetune lips color')
    parser.add_argument('--skin', action="store_true", default=False, help='whether to finetune foundation color')
    parser.add_argument('--eye', action="store_true", default=False, help='whether to finetune eye shadow color')
    parser.add_argument('--test_model', default='99_1260', type=str, help='which one to test')

    parser.add_argument('--img_src', type=str, help='src image')
    parser.add_argument('--img_ref', type=str, help='ref image')
    args = parser.parse_args()
    return args


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    if not requires_grad:
        return Variable(x, requires_grad=requires_grad)
    else:
        return Variable(x)

def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def test(args, config, G, img_src, img_ref, e=0, i=0):
    G_path = os.path.join(config.snapshot_path, '{}_G.pth'.format(config.test_model))
    G.load_state_dict(torch.load(G_path))
    G.eval()
    #time_total = time.time()
    time_total = 0
    start = time.time()
    img_A = img_src.expand(1, 3, img_src.size(2), img_src.size(2))
    img_B = img_ref.expand(1, 3, img_ref.size(2), img_ref.size(2))
    real_org = to_var(img_A)
    real_ref = to_var(img_B)

    image_list = []
    image_list_0 = []
    image_list.append(real_org)
    image_list.append(real_ref)
    image_list_0.append(real_org)
    image_list_0.append(real_ref)

    # Get makeup result
    fake_A, fake_B = G(real_org, real_ref)
    rec_B, rec_A = G(fake_B, fake_A)
    time_total += time.time() - start
    image_list.append(fake_A)
    image_list_0.append(fake_A)
    image_list.append(fake_B)
    image_list.append(rec_A)
    image_list.append(rec_B)

    image_list = torch.cat(image_list, dim=3)
    image_list_0 = torch.cat(image_list_0, dim=3)

    # result_path = config.vis_path + '_' + config.task_name
    img_src_name = args.img_src.split('/')[-1].split('.')[0]
    img_ref_name = args.img_ref.split('/')[-1].split('.')[0]
    result_path = os.path.join(config.vis_path, config.test_model)

    result_path_now = os.path.join(result_path, "multi")
    if not os.path.exists(result_path_now):
        os.makedirs(result_path_now)
    save_path = os.path.join(result_path_now, '{}_{}_fake.png'.format(img_src_name, img_ref_name))
    save_image(de_norm(image_list.data), save_path, nrow=1, padding=0, normalize=True)
    result_path_now = os.path.join(result_path, "single")

    if not os.path.exists(result_path_now):
        os.makedirs(result_path_now)
    save_path_0 = os.path.join(result_path_now, '{}_{}_single.png'.format(img_src_name, img_ref_name))
    save_image(de_norm(image_list_0.data), save_path_0, nrow=1, padding=0, normalize=True)
    print('Translated test images and saved into "{}"..!'.format(save_path_0))
    print("average time : {}".format(time_total))

if __name__ == '__main__':
    args = parse_args()
    print("Call with args:")
    print(args)
    config = merge_cfg_arg(config, args)

    config.test_model = args.test_model

    print("The config is:")
    print(config)

    # Create the directories if not exist
    if not os.path.exists(config.data_path):
        print("No datapath!!")

    dataset_config.dataset_path = os.path.join(config.data_path, args.data_path)

    # model
    g_conv_dim = config.g_conv_dim
    g_repeat_num = config.g_repeat_num
    G = net.Generator_branch(g_conv_dim, g_repeat_num)
    if torch.cuda.is_available():
        G = G.cuda()

    # src and ref images
    transform = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

    img_src = transform(Image.open(args.img_src).convert('RGB'))
    img_ref = transform(Image.open(args.img_ref).convert('RGB'))

    test(args, config, G, img_src, img_ref)
