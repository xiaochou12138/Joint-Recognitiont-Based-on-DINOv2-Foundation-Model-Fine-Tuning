import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class BasicDataset(Dataset):

    def __init__(self,patch_h,patch_w,datasetName,netType,train_mode = False):

        self.patch_h = patch_h
        self.patch_w = patch_w

        if netType == 'unet' or netType == 'deeplabv3plus':
            self.imgTrans = False
        else: 
            self.imgTrans = True

        self.transform = T.Compose([
            T.Resize((patch_h * 14, patch_w * 14)),
            T.ToTensor(),
        ])    

        self.dataset = datasetName

        if datasetName == 'fault':
            self.n1 = 224
            self.n2 = 224
            self.train_data_dir = 'data/synthetic_data/train/seis'
            self.train_label_dir = 'data/synthetic_data/train/fault'
            self.valid_data_dir = 'data/synthetic_data/valid/seis'
            self.valid_label_dir = 'data/synthetic_data/valid/fault'
        elif datasetName == 'horizon':
            self.n1 = 224
            self.n2 = 224
            self.train_data_dir = 'data/synthetic_data/train/seis_horizon'
            self.train_label_dir = 'data/synthetic_data/train/horizon'
            self.valid_data_dir = 'data/synthetic_data/valid/seis'
            self.valid_label_dir = 'data/synthetic_data/valid/horizon'
        elif datasetName == 'structure':
            self.n1 = 224
            self.n2 = 224
            self.train_data_dir = 'data/synthetic_data/train/seis1'
            self.train_label_dir = 'data/synthetic_data/train/label1'
            self.valid_data_dir = 'data/synthetic_data/valid/seis'
            self.valid_label_dir = 'data/synthetic_data/valid/label'
        else:
            print("Dataset error!!")
        print('netType:' + netType)
        print('dataset:' + datasetName)
        print('patch_h:' + str(patch_h))
        print('patch_w:' + str(patch_w))

        if train_mode:
            self.data_dir = self.train_data_dir
            self.label_dir = self.train_label_dir
        else:
            self.data_dir = self.valid_data_dir
            self.label_dir = self.valid_label_dir

        self.ids = len(os.listdir(self.data_dir))
        # self.ids = 50
    def __len__(self):
        return self.ids

    def __getitem__(self,index):
        
        dPath = self.data_dir+'/'+str(index)+'.dat'
        tPath = self.label_dir+'/'+str(index)+'.dat'
        data = np.fromfile(dPath,np.float32).reshape(self.n1,self.n2)
        label = np.fromfile(tPath,np.float32).reshape(self.n1,self.n2)

        data = np.reshape(data,(1,1,self.n1,self.n2))
        data = np.concatenate([data,self.data_aug(data)],axis=0)
        label = np.reshape(label,(1,1,self.n1,self.n2))
        label = np.concatenate([label,self.data_aug(label)],axis=0)

        data = (data - data.min())/(data.max() - data.min())

        data = data.repeat(3,axis=1)

        return data[:,:,:,:],label  # [:,:,:378, :126]

    def data_aug(self,data):
        b,c,h,w = data.shape
        data_fliplr = np.fliplr(np.squeeze(data))
        return data_fliplr.reshape(b,c,h,w)

if __name__ == '__main__':

    train_set = BasicDataset(16,16,'structure','setr1',True)
    print(train_set.__getitem__(0)[1].shape)
    print(len(train_set))
