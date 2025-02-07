import numpy as np
import torch
from dataset import MRIDataset
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from yAwareContrastiveLearning import yAwareCLModel
from losses import GeneralizedSupervisedNTXenLoss
from torch.nn import CrossEntropyLoss
from models.densenet import densenet121
from models.unet import UNet
import argparse
from config import Config, PRETRAINING, FINE_TUNING
from Mydata import MyDataset, CustomDataset

#从分类到回归，模型的最后num_class要改，loss函数要改，模型的finetuning函数要改，data的label要改


if __name__ == "__main__":

    # #设置随机种子数以复现结果
    seed = 42
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly !")
    args = parser.parse_args()  #解析命令行参数,解析后可以用args.属性名进行访问
    mode = PRETRAINING if args.mode == "pretraining" else FINE_TUNING  #这部分参数在控制台执行.py文件是通过--引入

    config = Config(mode)

    if config.mode == PRETRAINING:
        dataset_train = MRIDataset(config, training=True)
        dataset_val = MRIDataset(config, validation=True)
    else:
        ## Fill with your target dataset
        # dataset_train = Dataset()
        # dataset_val = Dataset()
        dataset_train = MyDataset(root_dir='H:/LSY/MySplit/Train', csv_dir='UCSF-PDGM-metadata_v2.csv')
        dataset_val = MyDataset(root_dir='H:/LSY/MySplit/Test', csv_dir='UCSF-PDGM-metadata_v2.csv')

    loader_train = DataLoader(dataset_train,
                              batch_size=config.batch_size,
                              sampler=RandomSampler(dataset_train),
                              collate_fn=dataset_train.collate_fn,
                              pin_memory=config.pin_mem,
                              num_workers=config.num_cpu_workers
                              )
    loader_val = DataLoader(dataset_val,
                            batch_size=config.batch_size,
                            sampler=RandomSampler(dataset_val),
                            collate_fn=dataset_val.collate_fn,
                            pin_memory=config.pin_mem,
                            num_workers=config.num_cpu_workers
                            )

    if config.mode == PRETRAINING:
        if config.model == "DenseNet":
            net = densenet121(mode="encoder", drop_rate=0.0)
        elif config.model == "UNet":
            net = UNet(config.num_classes, mode="simCLR")
        else:
            raise ValueError("Unkown model: %s"%config.model)
    else:
        if config.model == "DenseNet":
            net = densenet121(mode="classifier", drop_rate=0.0, num_classes=config.num_classes)
        elif config.model == "UNet":
            net = UNet(config.num_classes, mode="classif")
        else:
            raise ValueError("Unkown model: %s"%config.model)
    if config.mode == PRETRAINING:
        loss = GeneralizedSupervisedNTXenLoss(temperature=config.temperature,
                                              kernel='rbf',
                                              sigma=config.sigma,
                                              return_logits=True)
    elif config.mode == FINE_TUNING:
        loss = CrossEntropyLoss()
        # #给CE损失函数加上权重
        # weights = torch.FloatTensor([20, 20, 1]).cuda()
        # loss = CrossEntropyLoss(weight=weights)  # 交叉熵函数
        # #以上
        # loss = torch.nn.MSELoss() #用于回归任务
        # loss = torch.nn.L1Loss()

    model = yAwareCLModel(net, loss, loader_train, loader_val, config)

    if config.mode == PRETRAINING:
        model.pretraining()
    else:
        model.fine_tuning()
        # model.fine_tuning_regression() #用于回归任务




