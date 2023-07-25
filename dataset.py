from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from augmentations import Transformer, Crop, Cutout, Noise, Normalize, Blur, Flip
from Mydata import MyDataset

class MRIDataset(Dataset):

    def __init__(self, config, training=False, validation=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert training != validation
        #改：从Mydata中读取已写好的dataset
        self.custom_dataset = MyDataset(root_dir='I:/UCSF-PDGM-v3/PKG - UCSF-PDGM-v3/Mydeepmedic/UCSF-PDGM-v3',csv_dir='UCSF-PDGM-metadata_v2.csv')
        self.train_size = int(len(self.custom_dataset) * 0.7)
        self.test_size = len(self.custom_dataset) - self.train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.custom_dataset, [self.train_size, self.test_size])

        self.transforms = Transformer()
        self.config = config
        self.transforms.register(Normalize(), probability=1.0)
        #probability=1.0 表示这个数据转换操作将会在所有的样本上都被应用，即100%的概率进行转换。总结起来，这段代码的作用是将一个名为 Normalize() 的数据转换操作注册到 self.transforms 对象中，并指定这个转换操作在所有样本上都会被应用，即对数据集中的所有样本进行标准化操作。这样，当数据集中的样本被传递给模型进行训练时，Normalize() 操作将被自动应用，从而实现数据标准化的效果。

        if config.tf == "all_tf":
            self.transforms.register(Flip(), probability=0.5)
            self.transforms.register(Blur(sigma=(0.1, 1)), probability=0.5)
            self.transforms.register(Noise(sigma=(0.1, 1)), probability=0.5)
            self.transforms.register(Cutout(patch_size=np.ceil(np.array(config.input_size)/4)), probability=0.5)
            self.transforms.register(Crop(np.ceil(0.75*np.array(config.input_size)), "random", resize=True),
                                     probability=0.5)

        elif config.tf == "cutout":
            self.transforms.register(Cutout(patch_size=np.ceil(np.array(config.input_size)/4)), probability=1)

        elif config.tf == "crop":
            self.transforms.register(Crop(np.ceil(0.75*np.array(config.input_size)), "random", resize=True),
                                     probability=1)

        if training:
            #改：在之后getitem中处理数据问题
            # self.data = np.load(config.data_train)
            # self.labels = pd.read_csv(config.label_train)
            self.training = True

        elif validation:
            # self.data = np.load(config.data_val)
            # self.labels = pd.read_csv(config.label_val)
            self.validation = True

        # assert self.data.shape[1:] == tuple(config.input_size), "3D images must have shape {}".\
        #     format(config.input_size) #反斜杠是换行符，.format是用法，.\只是刚好巧了在一块

    def collate_fn(self, list_samples):
        list_x = torch.stack([torch.as_tensor(x, dtype=torch.float) for (x, y) in list_samples], dim=0)
        list_y = torch.stack([torch.as_tensor(y, dtype=torch.float) for (x, y) in list_samples], dim=0)

        return (list_x, list_y)

    def __getitem__(self, idx):
        if self.training:
            #在此处getitem处改代码使得能用上之前的代码
            data, labels = self.train_dataset[idx]
            # For a single input x, samples (t, t') ~ T to generate (t(x), t'(x))
            np.random.seed()
            x1 = self.transforms(self.data)
            x2 = self.transforms(self.data)
            # labels = self.labels[self.config.label_name].values[idx]
            x = np.stack((x1, x2), axis=0)
            return (x, labels)
        elif self.validation:
            data, labels = self.test_dataset[idx]
            # For a single input x, samples (t, t') ~ T to generate (t(x), t'(x))
            np.random.seed()
            x1 = self.transforms(self.data)
            x2 = self.transforms(self.data)
            # labels = self.labels[self.config.label_name].values[idx]
            x = np.stack((x1, x2), axis=0)
            return (x, labels)


    def __len__(self):
        if self.training:
            return len(self.train_dataset)
        elif self.validation:
            return len(self.test_dataset)
