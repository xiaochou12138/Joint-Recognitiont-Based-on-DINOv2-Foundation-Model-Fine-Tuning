import logging
import os
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(EXAMPLE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import BasicDataset
from models.adapter import dinov2_mla,dinov2_pup,dinov2_linear
from models.Dpt import dinov2_dpt
from models.unet import U_Net
import numpy as np
from tensorboardX import SummaryWriter
from loss.FocalLoss import Focal_Loss
from loss.DiceLoss import DiceLoss
from loss.WeightDiceLoss import WeightedDiceLoss
import random
import argparse
import loralib as lora

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# 计算梯度结构张量
def compute_structure_tensor(data):
    """
    计算地震数据在高度维度的横向梯度结构张量

    参数:
        data: 输入地震数据，形状为 (batch_size, channels, height, width)

    返回:
        结构张量，形状为 (batch_size, 3, height, width)
        三个通道分别对应: [gx², gx·gy, gy²]
    """
    if data.ndim != 4:
        raise ValueError("输入数据必须是4维张量 (batch, channels, height, width)")

    # 1. 计算高度和宽度方向的梯度
    # 使用Sobel滤波器增强梯度计算
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=data.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32, device=data.device).view(1, 1, 3, 3)

    # 扩展滤波器到所有通道
    sobel_x = sobel_x.repeat(data.size(1), 1, 1, 1)
    sobel_y = sobel_y.repeat(data.size(1), 1, 1, 1)

    # 计算梯度 (使用分组卷积保持通道独立性)
    gx = F.conv2d(data, sobel_x, padding=1, groups=data.size(1))
    gy = F.conv2d(data, sobel_y, padding=1, groups=data.size(1))

    # 2. 计算结构张量分量
    gx2 = gx.pow(2)  # ∂x²
    gy2 = gy.pow(2)  # ∂y²
    gx_gy = gx * gy  # ∂x·∂y

    # 3. 跨通道聚合 (对三个地震通道求和)
    J_xx = gx2.sum(dim=1, keepdim=True)  # 保持维度
    J_yy = gy2.sum(dim=1, keepdim=True)
    J_xy = gx_gy.sum(dim=1, keepdim=True)

    # 4. 组合结构张量
    structure_tensor = torch.cat([J_xx, J_xy, J_yy], dim=1)

    return structure_tensor[:,0:1,:,:]


def compute_second_derivatives(u):
    """
    计算输入数组的二阶导数 uxx 和 uyy

    参数:
        u: 输入数组，形状为 (batch, channels, height, width)

    返回:
        uxx: 二阶导数 (x方向)
        uyy: 二阶导数 (y方向)
    """
    if u.ndim != 4:
        raise ValueError("输入数组必须是4维张量 (batch, channels, height, width)")

    # 定义二阶导数卷积核
    kernel_xx = torch.tensor([[0, 0, 0],
                              [1, -2, 1],
                              [0, 0, 0]], dtype=torch.float32, device=u.device)

    kernel_yy = torch.tensor([[0, 1, 0],
                              [0, -2, 0],
                              [0, 1, 0]], dtype=torch.float32, device=u.device)

    # 调整核形状以适应卷积操作 (out_channels, in_channels, H, W)
    kernel_xx = kernel_xx.view(1, 1, 3, 3)
    kernel_yy = kernel_yy.view(1, 1, 3, 3)

    # 计算二阶导数
    uxx = F.conv2d(u, kernel_xx, padding=1)
    uyy = F.conv2d(u, kernel_yy, padding=1)

    return uxx, uyy

def Pde_horizon(sample, net):
    # 1、计算地震数据sample的梯度结构张量
    sample = sample.to(args.device)
    st_seis = compute_structure_tensor(sample)
    st_seis = (st_seis - st_seis.min())/(st_seis.max()-st_seis.min()).float().to(args.device)

    # 2、处理地震层位预测结果，将其转为类似RGT形式
    # 找到层位点的位置
    pre_horizon = net(sample, (args.n1,args.n2))
    RGT_arr = torch.zeros(size=(pre_horizon.shape[0], pre_horizon.shape[1], pre_horizon.shape[2], pre_horizon.shape[3]))
    # 保留 >0.5 的值，其余置 0
    pre_horizon = torch.where(pre_horizon > torch.tensor(0.04), pre_horizon, torch.tensor(0.0))
    for i in range(len(pre_horizon)):
        pre_horizon1 = pre_horizon[i][0]
        for j in range(len(pre_horizon1)):
            pre_horizon2 = pre_horizon1[j]
            # 找到单道数据层位点
            idx = torch.where(pre_horizon2!=0)
            arr = 1
            for k in range(len(idx[0])):
                idx1 = idx[0][k]
                if(k==0):
                    RGT_arr[i,0,j, 0:idx1] = arr
                else:
                    RGT_arr[i,0,j, idx[0][k-1]:idx1] = arr
                arr = arr + 1
    RGT_arr = torch.where(RGT_arr==torch.tensor(0), torch.tensor(1), RGT_arr)
    # 3、计算RGT的梯度结构张量
    st_RGT = compute_structure_tensor(RGT_arr.to(args.device))
    st_RGT = (st_RGT - st_RGT.min())/(st_RGT.max()-st_RGT.min()).float().to(args.device)

    return nn.MSELoss()(st_seis, st_RGT), RGT_arr

def cal_input_fault(sample, RGT):
    # 计算RGT的二阶导 二阶导有助于突出地震剖面中大尺度的不连续特征，由于uxx uyy聚焦于全局构造变化 低阶特征强化了局部反射细节，两则结合可以在噪声以及断层规模小的情况下，依旧可以识别到连续且合理的断层走向
    RGT_xx, RGT_yy = compute_second_derivatives(RGT)
    sample_channel1 = sample[:,0:1,:,:]
    return torch.cat((sample_channel1,RGT_xx,RGT_yy),dim=1)


def main(args,logger):
    dir_checkpoint = 'checkpoint_structure_task/' + args.dataset + "/" + args.loss + "/" +args.netType
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    if args.dataset == 'fault':
        args.n1, args.n2 = 224, 224
        args.classes = 1
        args.patch_h = 16
        args.patch_w = 16
        args.batch_size = 3
    elif args.dataset == 'horizon':
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
    elif args.netType == "mla":
        net = dinov2_mla(args.classes, pretrain=args.dpt, vit_type=args.vt,frozen=frozen,finetune_method=args.checkpointName)
    elif args.netType == "pup":
        net = dinov2_pup(args.classes, pretrain=args.dpt, vit_type=args.vt,frozen=frozen,finetune_method=args.checkpointName)
    elif args.netType == "dpt":
        net = dinov2_dpt(args.classes, pretrain=args.dpt, vit_type=args.vt,frozen=frozen,finetune_method=args.checkpointName)

    logger.info(f'\t{args.netType} NetWork:\n'
                 f'\t{args.classes } num classes\n'
                 f'\t{args.dataset} dataset\n'
                 f'\t{args.vt} vitType\n'
                 f'\t{args.loss} loss\n')
    goTrain(args,
            dir_checkpoint,
            net=net,
            patch_h = args.patch_h,
            patch_w = args.patch_w,
            epochs=args.epochs,
            batch_size= int(args.batch_size),
            learning_rate= args.lr,
            num_classes = args.classes,
            save_checkpoint=args.save_checkpoint
                )
def goTrain(args,
            dir_checkpoint,
            net,
            patch_h,
            patch_w,
            num_classes : int,
            epochs:int = 5,
            batch_size: int = 1,
            learning_rate: float = 1e-4,
            save_checkpoint: bool = True):

    net.to(device)
    get_parameter_number(net)

    # Create dataset
    train_set = BasicDataset(patch_h, patch_w, args.dataset,args.netType, train_mode=True)
    valid_set = BasicDataset(patch_h, patch_w, args.dataset,args.netType, train_mode=False)

    #Create data loaders
    train_loader= DataLoader(dataset = train_set,batch_size = batch_size, shuffle=True)
    valid_loader= DataLoader(dataset = valid_set,batch_size = batch_size, shuffle=False)

    logger.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_set)}
        Validation size: {len(valid_set)}
        Checkpoints:     {save_checkpoint}
    ''')

    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=0.01,betas=[0.7,0.999])
    if args.al:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "focal":
        criterion = Focal_Loss(args.classes,device=args.device)
    elif args.loss == "dice":
        criterion = DiceLoss(args.classes)
    elif args.loss == "wdice":
        criterion = WeightedDiceLoss(args.classes,device=args.device)
    elif args.loss == "bace":
        if args.dataset == "seam":
            weight = torch.tensor([1.216,0.395,3.673,0.573,14.193,1.798]).reshape(-1,1).to(args.device)
            criterion = nn.CrossEntropyLoss(weight=weight)

    #Tensorboard open
    writer = SummaryWriter('Tensorboard/'+args.dataset+'/' + args.loss + '/')

    # Begin training
    train_loss = []
    valid_loss=[]
    train_pa = []
    valid_pa = []
    MinTrainLoss = 1e7
    MinValidLoss = 1e7
    net.train()
    warmup_steps = 10
    ini_lr = learning_rate*10
    for epoch in range(1,epochs+1):
        if args.al=="True":
            if epoch < warmup_steps:
                warmup_percent_done = epoch/warmup_steps
                optimizer.param_groups[0]['lr'] = ini_lr * warmup_percent_done
            else:
                scheduler.step()
        total_train_loss = []
        total_valid_loss = []
        with tqdm(total = len(train_set),desc=f'Epoch {epoch}/{epochs}',unit = 'img') as t:
            for data,label in train_loader:
                b1,b2,c,h,w = data.shape
                data = data.to(device).reshape(b1*b2,c,h,w)
                b1,b2,c,h,w = label.shape
                label = label.to(device).reshape(b1*b2,c,h,w)
                optimizer.zero_grad()
                outputs = net(data,(args.n1,args.n2))
                if args.loss == "bce":
                    loss = criterion(outputs,label.float())
                else:
                    loss = criterion(outputs,label.float())

                # # Pinn 损失 梯度结构张量
                # Pinn_loss, RGT_arr= Pde_horizon(data, net)
                # loss = loss + 0.005*Pinn_loss
                #
                # fault_input = cal_input_fault(data, RGT_arr.to(device))

                loss.backward()
                optimizer.step()
                t.update(batch_size)
                t.set_postfix(**{'train_loss': loss.item(),'lr': optimizer.param_groups[0]['lr']})
                total_train_loss.append(loss.item())
            train_loss.append(np.mean(total_train_loss))
            logger.info(f"Epoch {epoch} - TrainSet - Loss: {train_loss[-1]}")

        if train_loss[-1]<MinTrainLoss:
            torch.save(net.state_dict(), dir_checkpoint + "/"+args.checkpointName + "_" + args.vt+"_minloss_train.pth")
            if args.checkpointName=="lora":
                torch.save(lora.lora_state_dict(net), dir_checkpoint + "/"+args.checkpointName + "_" + args.vt+"_minloss_train_lora.pth")
            MinTrainLoss = train_loss[-1]
            logger.info(f'min_train_loss saved!')

        net.eval()
        with tqdm(total = len(valid_set),desc=f'Epoch {epoch}/{epochs}',unit = 'img') as t:
            for data,label in valid_loader:
                b1,b2,c,h,w = data.shape
                data = data.to(device).reshape(b1*b2,c,h,w)
                b1,b2,c,h,w = label.shape
                label = label.to(device).reshape(b1*b2,c,h,w)
                optimizer.zero_grad()
                outputs = net(data,(args.n1,args.n2))
                if args.loss == "bce":
                    loss = criterion(outputs,label.float())
                else:
                    loss = criterion(outputs,label.float())
                optimizer.step()
                t.update(batch_size)
                t.set_postfix(**{'valid_loss': loss.item(),'lr': optimizer.param_groups[0]['lr']})
                total_valid_loss.append(loss.item())
            valid_loss.append(np.mean(total_valid_loss))
            logger.info(f"Epoch {epoch} - ValidateSet - Loss: {valid_loss[-1]}")

        if valid_loss[-1]<MinValidLoss:
            torch.save(net.state_dict(), dir_checkpoint + "/"+args.checkpointName + "_" + args.vt+"_minloss_valid.pth")
            if args.checkpointName=="lora":
                torch.save(lora.lora_state_dict(net), dir_checkpoint + "/"+args.checkpointName + "_" + args.vt+"_minloss_valid_lora.pth")
            MinValidLoss = valid_loss[-1]
            logger.info(f'min_valid_loss saved!')
        #Tensorboard writting
        writer.add_scalars('loss_' + args.netType + '_' +args.checkpointName + "_" + args.vt,{'train':train_loss[epoch-1], 'valid':valid_loss[epoch-1]},epoch)
        writer.add_scalars('lr_' + args.netType + '_' +args.checkpointName + "_" + args.vt,{'lr':optimizer.param_groups[0]['lr']},epoch)

    #Tensorboard close
    writer.close()

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Model Total: %d'%total_num)
    logger.info('Model Trainable: %d'%trainable_num)

def get_args():
    parser = argparse.ArgumentParser(description='train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Epochs')
    parser.add_argument('--learning_rate', '-l', dest='lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--loss', '-loss',  type=str, default='mse')
    parser.add_argument('--anneal_lr', '-a', dest='al', type=str, default="False")
    parser.add_argument('--dpt', '-p', type=str, default="True", help='dinov2 pretrain')
    parser.add_argument('--vt', '-v', type=str, default="small")
    parser.add_argument('--checkpointName', '-cp',  type=str, default='lora', help='lora or unfrozen')
    parser.add_argument('--netType', '-net',  type=str, default='pup', help='dpt,pup,mla,linear,unet,deeplabv3plus')
    parser.add_argument('--dataset', '-d', type=str, default='structure')
    parser.add_argument('--device', '-dn', type=str, default='cuda:0')
    parser.add_argument('--save_checkpoint', '-checkpoint', type=bool, default=True)
    return parser.parse_args()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args()
    device = args.device

    log_dir = "log/" + args.dataset + "/" + args.loss + "/" + args.netType
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{args.checkpointName}_{args.vt}.txt")
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info(f"{args.dataset}/{args.loss}/{args.netType}/{args.checkpointName}_{args.vt}")
    logger = logging.getLogger()
    main(args,logger)
