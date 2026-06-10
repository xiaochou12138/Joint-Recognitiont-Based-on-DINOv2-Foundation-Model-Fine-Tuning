from tqdm import tqdm
import os
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(EXAMPLE_DIR)

import torch
from torch.utils.data import DataLoader
from dataset import BasicDataset
from models.adapter import dinov2_mla, dinov2_pup, dinov2_linear
from models.unet import U_Net
from models.Dpt import dinov2_dpt
import matplotlib
matplotlib.use('TKAgg')  # 切换到非交互式后端
import matplotlib.pyplot as plt
import numpy as np
# from demo_classification import get_args
from train import get_args

def goPredict(net, device, patch_h, patch_w, mPath, loraPath, datasetname, netType):
    if os.path.exists(mPath):
        if args.checkpointName=="lora":
            net.load_state_dict(torch.load(mPath, map_location=device),strict=True)
            net.load_state_dict(torch.load(loraPath, map_location=device),strict=False)
        else:
            net.load_state_dict(torch.load(mPath, map_location=device),strict=True)
    net.to(device)
    net.eval()
    valid_set = BasicDataset(patch_h, patch_w, datasetname,netType,False)
    valid_loader = DataLoader(dataset = valid_set,batch_size = args.batch_size, shuffle=False)
    data_list = []
    predict = []
    target = []

    for data, label in valid_loader:
        b1,b2,c,h,w = data.shape
        data = data.to(device).reshape(b1*b2,c,h,w)
        b1,b2,c,h,w = label.shape
        label = label.to(device).reshape(b1*b2,c,h,w)
        preds = net(data, (args.n1, args.n2))

        predict.append(np.squeeze(preds.detach().cpu().numpy()).astype(np.float32))
        data_list.append(np.squeeze(data[:,0:1,:,:].detach().cpu().numpy()).astype(np.float32))
        target.append(np.squeeze(label.detach().cpu().numpy()).astype(np.float32))

    return data_list, predict, target


if __name__ == '__main__':
    args = get_args()
    device = args.device

    if args.dataset == 'horizon':
        args.n1, args.n2 = 224, 224
        args.classes = 1
        args.patch_h = 16
        args.patch_w = 16
        args.batch_size = 3
    elif args.dataset == 'fault':
        args.n1, args.n2 = 224, 224
        args.classes = 1
        args.patch_h = 16
        args.patch_w = 16
        args.batch_size = 3
    elif args.dataset == 'structure':
        args.n1, args.n2 = 224, 224
        args.classes = 1
        args.patch_h = 16
        args.patch_w = 16
        args.batch_size = 3

    if args.checkpointName in ["unfrozen","lora"]:
        frozen = False
    elif args.checkpointName == "frozen":
        frozen = True

    if args.netType == "unet":
        net = U_Net(3,args.classes)
    elif args.netType == "linear":
        net = dinov2_linear(args.classes, pretrain=args.dpt, vit_type=args.vt,frozen=frozen,finetune_method=args.checkpointName)
    elif args.netType == "pup":
        net = dinov2_pup(args.classes, pretrain=args.dpt, vit_type=args.vt,frozen=frozen,finetune_method=args.checkpointName)
    elif args.netType == "mla":
        net = dinov2_mla(args.classes, pretrain=args.dpt, vit_type=args.vt,frozen=frozen,finetune_method=args.checkpointName)
    elif args.netType == "dpt":
        net = dinov2_dpt(args.classes, pretrain=args.dpt, vit_type=args.vt,frozen=frozen,finetune_method=args.checkpointName)

    model_Path = 'checkpoint_structure_task/'+ args.dataset +'/'+ args.loss +'/'+args.netType +'/'+args.checkpointName + "_" + args.vt+'_minloss_valid.pth'
    lora_Path = 'checkpoint_structure_task/'+ args.dataset +'/'+ args.loss +'/'+args.netType +'/'+args.checkpointName + "_" + args.vt+'_minloss_valid_lora.pth'
    pngPath = 'png/'+ args.dataset +'/'+ args.netType +'/'+ args.loss +'/' +args.checkpointName + "_" + args.vt+'/'

    if not os.path.exists(pngPath):
        os.makedirs(pngPath)

    predictPath = os.path.join(pngPath,'prediction')
    if not os.path.exists(predictPath):
        os.makedirs(predictPath)

    patch_h = args.patch_h
    patch_w = args.patch_w

    #paint
    cmin = 0
    cmax = args.classes - 1

    data, predict, target = goPredict(net, device, patch_h, patch_w, model_Path, lora_Path, args.dataset, args.netType)
    for i in tqdm(range(len(data))):
        # predict[i][0:3].tofile(predictPath+'/prediction_unet_'+str(i+1)+'.dat')
        plt.figure(figsize=[10,10])
        plt.imshow(data[i][0],  cmap='seismic')
        plt.show()

        plt.figure(figsize=[10,10])
        plt.imshow(predict[i][0], cmap='RdBu')
        plt.show()

        plt.figure(figsize=[10,10])
        plt.imshow(target[i][0], cmap='RdBu')
        plt.show()
