import os
import torch
from torch.nn import DataParallel
from tqdm import tqdm
import logging
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score



class yAwareCLModel:

    def __init__(self, net, loss, loader_train, loader_val, config, scheduler=None):
        """

        Parameters
        ----------
        net: subclass of nn.Module
        loss: callable fn with args (y_pred, y_true)
        loader_train, loader_val: pytorch DataLoaders for training/validation
        config: Config object with hyperparameters
        scheduler (optional)
        """
        super().__init__()
        self.logger = logging.getLogger("yAwareCL")
        self.loss = loss
        self.model = net
        self.optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = scheduler
        self.loader = loader_train
        self.loader_val = loader_val
        self.device = torch.device("cuda" if config.cuda else "cpu")
        if config.cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found: set cuda=False parameter.")
        self.config = config
        self.metrics = {}

        if hasattr(config, 'pretrained_path') and config.pretrained_path is not None:

            # self.load_model(config.pretrained_path)
            self.load_and_freeze_model(config.pretrained_path) #冻结预训练参数
            for k, v in self.model.named_parameters():
                print('{}: {}'.format(k, v.requires_grad))

        #hasattr(config, 'pretrained_path'): 这是Python的内置函数 hasattr()，用于检查一个对象（这里是 config 对象）是否有指定的属性。在这里，它用于检查 config 对象是否具有名为 pretrained_path 的属性。

        self.model = DataParallel(self.model).to(self.device)
        #DataParallel(self.model): 这是 PyTorch 中的 DataParallel 类，用于实现数据并行处理。数据并行是一种将数据分割并在多个设备上并行处理的技术，用于加速训练过程。DataParallel 可以在多个 GPU 上并行处理数据，并将梯度的计算和参数更新同步。

    def pretraining(self):
        print(self.loss)
        print(self.optimizer)

        for epoch in range(self.config.nb_epochs):

            ## Training step
            self.model.train()
            nb_batch = len(self.loader)
            training_loss = 0
            pbar = tqdm(total=nb_batch, desc="Training")
            for (inputs, labels) in self.loader:
                pbar.update()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                z_i = self.model(inputs[:, 0, :])
                z_j = self.model(inputs[:, 1, :])
                batch_loss, logits, target = self.loss(z_i, z_j, labels)
                batch_loss.backward()
                self.optimizer.step()
                training_loss += float(batch_loss) / nb_batch
            pbar.close()

            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation")
            val_loss = 0
            val_values = {}
            with torch.no_grad():
                self.model.eval()
                for (inputs, labels) in self.loader_val:
                    pbar.update()
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    z_i = self.model(inputs[:, 0, :])
                    z_j = self.model(inputs[:, 1, :])
                    batch_loss, logits, target = self.loss(z_i, z_j, labels)
                    val_loss += float(batch_loss) / nb_batch
                    for name, metric in self.metrics.items():
                        if name not in val_values:
                            val_values[name] = 0
                        val_values[name] += metric(logits, target) / nb_batch
            pbar.close()

            metrics = "\t".join(["Validation {}: {:.4f}".format(m, v) for (m, v) in val_values.items()])
            print("Epoch [{}/{}] Training loss = {:.4f}\t Validation loss = {:.4f}\t".format(
                epoch+1, self.config.nb_epochs, training_loss, val_loss)+metrics, flush=True)

            if self.scheduler is not None:
                self.scheduler.step()

            if (epoch % self.config.nb_epochs_per_saving == 0 or epoch == self.config.nb_epochs - 1) and epoch > 0:
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()},
                    os.path.join(self.config.checkpoint_dir, "{name}_epoch_{epoch}.pth".
                                 format(name="y-Aware_Contrastive_MRI", epoch=epoch)))


    def fine_tuning(self):
        print(self.loss)
        print(self.optimizer)
        best_loss = 100 #设定初始best loss以保存最佳参数模型

        for epoch in range(self.config.nb_epochs):
            ## Training step
            self.model.train()
            nb_batch = len(self.loader) #self.loader = train_loader
            # training_loss = []
            training_loss = 0


            #从此处开始修改代码以查看训练和验证的准确度
            y_true = []
            y_pred = []
            #以上

            pbar = tqdm(total=nb_batch, desc="Training")
            #这段代码使用了 tqdm 库中的 tqdm 函数，用于在终端显示一个进度条，以跟踪代码执行的进度。

            for (inputs, labels) in self.loader:
                pbar.update() #pbar.update() 方法来更新进度条的进度。例如，如果代码中有一个循环进行了 nb_batch 次迭代，那么在每次迭代中，调用 pbar.update(1) 将进度条前进一步，最终达到总步数 nb_batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                y = self.model(inputs)

                batch_loss = self.loss(y,labels.long()) #这里加了labels后.long()防止出现浮点数错误,long() 函数将数字或字符串转换为一个长整型

                batch_loss.backward()
                self.optimizer.step()
                training_loss += float(batch_loss) / nb_batch

                #查看准确率,F1分数代码
                _, predicted = torch.max(y.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                #以上

            #查看准确率,F1分数代码
            train_f1 = f1_score(y_true, y_pred, average=None)
            train_acc = accuracy_score(y_true, y_pred)
            #以上
            pbar.close()

            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation")
            val_loss = 0

            # 从此处开始修改代码以查看训练和验证的准确度
            y_true = []
            y_pred = []
            # 以上

            with torch.no_grad():
                self.model.eval()
                for (inputs, labels) in self.loader_val:
                    pbar.update()
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    y = self.model(inputs)

                    batch_loss = self.loss(y,labels.long()) #这里加了labels后.long()防止出现浮点数错误

                    val_loss += float(batch_loss) / nb_batch

                    # 查看准确率,F1分数代码
                    _, predicted = torch.max(y.data, 1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
                    # 以上

            #查看准确率,F1分数代码
            val_f1 = f1_score(y_true, y_pred, average=None)
            val_acc = accuracy_score(y_true, y_pred)
            #以上
            pbar.close()

            #用于分类任务
            print("Epoch [{}/{}] Training loss = {:.4f}\t Validation loss = {:.4f}\t Training ACC = {:.4f}\t Validation ACC = {:.4f}\t".format(
                epoch+1, self.config.nb_epochs, training_loss, val_loss, train_acc, val_acc), flush=True)
            print('Traing F1:', train_f1)
            print('Validation F1:', val_f1)
            #以上

            save = True
            save_path = 'Result/freeze_ASL_Regressopm_AllSamples_Epoch{}.pt'.format(epoch+1)
            #保存最佳权重
            if val_loss < best_loss:
                best_loss = val_loss  # 更新最高精确度
                if save:
                    torch.save(self.model.state_dict(), save_path)  # 保存当前最佳权重参数
                    print('Save Epoch[{}] to the save path'.format(epoch+1))

            if self.scheduler is not None:
                self.scheduler.step()

    def fine_tuning_regression(self):
        print(self.loss)
        print(self.optimizer)

        for epoch in range(self.config.nb_epochs):
            ## Training step
            self.model.train()
            nb_batch = len(self.loader) #self.loader = train_loader
            training_loss = 0
            pbar = tqdm(total=nb_batch, desc="Training")
            #这段代码使用了 tqdm 库中的 tqdm 函数，用于在终端显示一个进度条，以跟踪代码执行的进度。

            for (inputs, labels) in self.loader:
                pbar.update() #pbar.update() 方法来更新进度条的进度。例如，如果代码中有一个循环进行了 nb_batch 次迭代，那么在每次迭代中，调用 pbar.update(1) 将进度条前进一步，最终达到总步数 nb_batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                y = self.model(inputs)
                batch_loss = self.loss(y, labels)
                batch_loss.backward()
                self.optimizer.step()
                training_loss += float(batch_loss) / nb_batch

            pbar.close()

            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation")
            val_loss = 0

            with torch.no_grad():
                self.model.eval()
                for (inputs, labels) in self.loader_val:
                    pbar.update()
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    y = self.model(inputs)

                    batch_loss = self.loss(y, labels)
                    val_loss += float(batch_loss) / nb_batch
            pbar.close()

            #用于回归任务
            print("Epoch [{}/{}] Training loss = {:.4f}\t Validation loss = {:.4f}\t".format(epoch + 1, self.config.nb_epochs, training_loss, val_loss), flush=True)


            if self.scheduler is not None:
                self.scheduler.step()


    def load_model(self, path):
        checkpoint = None
        try:
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        except BaseException as e:
            self.logger.error('Impossible to load the checkpoint: %s' % str(e))
        if checkpoint is not None:
            try:
                if hasattr(checkpoint, "state_dict"):
                    unexpected = self.model.load_state_dict(checkpoint.state_dict())
                    self.logger.info('Model loading info: {}'.format(unexpected))
                elif isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        unexpected = self.model.load_state_dict(checkpoint["model"], strict=False)
                        self.logger.info('Model loading info: {}'.format(unexpected))
                else:
                    unexpected = self.model.load_state_dict(checkpoint)
                    self.logger.info('Model loading info: {}'.format(unexpected))
            except BaseException as e:
                raise ValueError('Error while loading the model\'s weights: %s' % str(e))

    def load_and_freeze_model(self, path): #冻结加载的预训练参数
        checkpoint = None
        try:
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        except BaseException as e:
            self.logger.error('Impossible to load the checkpoint: %s' % str(e))
        if checkpoint is not None:
            try:
                if hasattr(checkpoint, "state_dict"):
                    unexpected = self.model.load_state_dict(checkpoint.state_dict())
                    self.logger.info('Model loading info: {}'.format(unexpected))
                elif isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        unexpected = self.model.load_state_dict(checkpoint["model"], strict=False)
                        self.logger.info('Model loading info: {}'.format(unexpected))
                else:
                    unexpected = self.model.load_state_dict(checkpoint)
                    self.logger.info('Model loading info: {}'.format(unexpected))

                # 冻结加载的预训练参数
                for name, param in self.model.named_parameters():
                    if "features" in name:
                        param.requires_grad = False
                        print(name)


            except BaseException as e:
                raise ValueError('Error while loading the model\'s weights: %s' % str(e))
            #pull 测试



