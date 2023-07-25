import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import pandas as pd
from dataset import MRIDataset
from torch.utils.data import Subset

class MyDataset(MRIDataset):
    def __init__(self, root_dir, csv_dir, transform=None):
        self.root_dir = root_dir
        self.samples = os.listdir(root_dir)
        self.csv_file = pd.read_csv(csv_dir)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.root_dir, self.samples[idx])

        # Load each MRI sequence from the nii files,-6是除去文件夹名称里的_nifti
        t1 = nib.load(os.path.join(sample_path, self.samples[idx][:-6] + '_T1.nii.gz')).get_fdata()
        t1c = nib.load(os.path.join(sample_path, self.samples[idx][:-6] + '_T1c.nii.gz')).get_fdata()
        t2 = nib.load(os.path.join(sample_path, self.samples[idx][:-6] + '_T2.nii.gz')).get_fdata()
        flair = nib.load(os.path.join(sample_path, self.samples[idx][:-6] + '_FLAIR.nii.gz')).get_fdata()
        asl = nib.load(os.path.join(sample_path, self.samples[idx][:-6] + '_ASL.nii.gz')).get_fdata()

        # Combine the four MRI sequences into a single input tensor
        # input_tensor = torch.Tensor(np.stack([t1, t1c, t2, flair, asl], axis=0))
        input_tensor = torch.Tensor(t1)
        # input_tensor = input_tensor.permute(0,3,1,2) #将维度重排为 [C, D, H, W]，有了transform就不需要在这里重排了，这个代码要求数据格式为[C, H, W, D]这里就不进行permute了
        if self.transform:
            input_tensor = self.transform(input_tensor)

        # 从csv文件中读取label
        # 原ID比csv文件中的ID多了个0，比如UCSF-PDGM-0004，csv中是UCSF-PDGM-004。所以要修改一下
        id = self.samples[idx][:-6]
        id_fit = id[0:-4] + id[-3:]
        label = self.csv_file.loc[self.csv_file['ID'] == id_fit, 'WHO CNS Grade'].values[0]
        label = label - 2  # 从[2,3,4]转为[0,1,2]
        # PyTorch会自动把整数型的label转为one-hot型，用于计算CE loss这里需要确保label是从0开始的,from深入浅出pytorch

        # Return the input tensor and any additional labels or targets
        return input_tensor, label  # label是你的样本的标签，需要自己定义


class CustomDataset(MyDataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def collate_fn(self, list_samples):
        list_x = torch.stack([torch.as_tensor(x, dtype=torch.float) for (x, y) in list_samples], dim=0)
        list_y = torch.stack([torch.as_tensor(y, dtype=torch.float) for (x, y) in list_samples], dim=0)

        return (list_x, list_y)