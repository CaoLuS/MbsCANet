"""
    @Author: Panke
    @Time: 2022-11-08  20:36
    @Email: None
    @File: Train.py
    @Project: MbsCANet
"""

import argparse
import configparser
import os
import torch
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from src.utils.SaveData import SaveData
import matplotlib.pyplot as plt

from src.utils.BreastDataScanner import DataScanner
import src.utils.GetVariousPath as path_utils
from src.models.MbsCANet import MbsCANet

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.benchmark = True


class BCNNManager(object):
    def __init__(self, options):
        print("Prepare network and data")
        self._options = options
        # 调用网络，记得设置
        self.model = MbsCANet()  # 定义网络模型
        # 记得设置
        self._magnification = '40x'
        num_params = self.count_parameters(self.model)
        print('params is :', num_params)

        self._net = torch.nn.DataParallel(self.model).cuda()
        # 损失函数
        self._criterion = torch.nn.CrossEntropyLoss().cuda()

        # 优化器
        self._optimizer = torch.optim.SGD(self._net.parameters(),
                                          lr=self._options['base_lr'],
                                          momentum=0.9,
                                          weight_decay=self._options['weight_decay'])
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer,
                                                                     mode='max',
                                                                     factor=0.1,
                                                                     patience=8,
                                                                     verbose=True,
                                                                     threshold=0.0001)
        # self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=self._options['base_lr'])
        data_transforms = {
            'train': transforms.Compose([transforms.Resize(size=(224, 224)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.ToTensor(),
                                         # transforms.Normalize([0.8043304, 0.6528069, 0.77411586],
                                         #                      [0.08765075, 0.11498511, 0.08631402]),  # 40
                                         # transforms.Normalize([0.79453534, 0.6345977, 0.7710843],
                                         #                      [0.105881795, 0.13931349, 0.09464326]),  # 100
                                         # transforms.Normalize([0.78760445, 0.62296575, 0.7685142],
                                         #                      [0.1035481, 0.13547403, 0.08270253]),  # 200
                                         # transforms.Normalize([0.75536066, 0.5881801, 0.74234426],
                                         #                      [0.117991276, 0.15735142, 0.08764123]),  # 400
                                         # transforms.RandomErasing()
                                         ]),
            'validate': transforms.Compose([transforms.Resize(size=(224, 224)),
                                            transforms.ToTensor(),
                                            # transforms.Normalize([0.8043304, 0.6528069, 0.77411586],
                                            #                      [0.08765075, 0.11498511, 0.08631402]),  # 40
                                            # transforms.Normalize([0.79453534, 0.6345977, 0.7710843],
                                            #                      [0.105881795, 0.13931349, 0.09464326]),  # 100
                                            # transforms.Normalize([0.78760445, 0.62296575, 0.7685142],
                                            #                      [0.1035481, 0.13547403, 0.08270253]),  # 200
                                            # transforms.Normalize([0.75536066, 0.5881801, 0.74234426],
                                            #                      [0.117991276, 0.15735142, 0.08764123]),  # 400
                                            # transforms.RandomErasing()
                                            ])
        }

        dataset_path = path_utils.get_dataset_magnification_path('data', 'Breakhis', self._magnification)

        train_dataset = DataScanner(data_path=dataset_path, for_what='train',
                                    transform=data_transforms['train'], download=False)

        validate_dataset = DataScanner(data_path=dataset_path, for_what='validate',
                                       transform=data_transforms['validate'], download=False)

        self._train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self._options['batch_size'],
                                                         shuffle=True, num_workers=4, pin_memory=False)

        self._validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=64,
                                                            shuffle=False, num_workers=4, pin_memory=False)

    def count_parameters(self, model: object) -> object:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params / 1000000

    def train(self):
        """
        训练网络
        :return: None
        """
        # 清空pth文件夹下边的所有文件
        # DirHandler.clear_dir('pth')
        print('Start training')
        self._net.train()
        best_acc = 0.0
        best_epoch = None
        print('Epoch \t Train loss \t Validate loss \t Train acc \t Test acc \t Time')
        lr_list = []
        loss_list = []
        # 保存所有训练的数据和参数
        data_list = []
        for epoch in range(self._options['epochs']):
            if epoch >= 5:
                if epoch % 5 == 0:
                    for p in self._optimizer.param_groups:
                        self._options['base_lr'] *= 0.5
                        print('学习率:\t{}'.format(self._options['base_lr']))
                lr_list.append(self._options['base_lr'])

            epoch_loss = []
            num_correct = 0
            num_total = 0
            tic = time.time()

            for images, labels in self._train_loader:
                images = images.cuda()
                labels = labels.cuda()

                # 前向传播
                score = self._net(images)
                loss = self._criterion(score, labels)

                with torch.no_grad():
                    epoch_loss.append(loss.item())
                    prediction = torch.argmax(score, dim=1)
                    num_total += labels.size(0)
                    num_correct += torch.sum(prediction == labels).item()

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                del images, labels, score, loss, prediction

            train_acc = 100 * num_correct / num_total
            validate_acc, validate_loss = self._validate_accuracy(self._validate_loader)

            if validate_acc >= best_acc:
                best_acc = validate_acc
                best_epoch = epoch + 1
                print('*', end='')
            if epoch >= 29:
                save_path = path_utils.get_save_pth_file('pth', 'bcnn_%s_epoch_%d.pth' % ('all', epoch + 1))
                torch.save(self._net.state_dict(), save_path)
            toc = time.time()
            train_loss = sum(epoch_loss) / len(epoch_loss)
            train_time = (toc - tic) / 60
            # 将每次的训练数据和参数保存到一个dict中
            data_dict = {'epoch': epoch + 1,
                         'train_loss': train_loss,
                         'train_acc': train_acc,
                         'validate_acc': validate_acc,
                         'time': train_time}
            loss_list.append(train_loss)
            # 将上边的dict添加到data_list中
            data_list.append(data_dict)
            # 打印训练数据和参数
            print('%d\t%4.3f\t\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f min' % (
                epoch + 1, train_loss, validate_loss, train_acc, validate_acc, train_time))
            self._scheduler.step(validate_acc)
            # self._scheduler.step()
        # 将训练的参数和数据保存到Excel和Json文件中
        SaveData.save_train_parameters(self.model._get_name(), self._magnification, data_list)
        print("Best at epoch %d, validate accuracy %4.2f" % (best_epoch, best_acc))
        print('Model name is : {}'.format(self.model._get_name()))
        x = list(range(len(loss_list)))
        plt.plot(x, loss_list)
        plt.show()

    def _validate_accuracy(self, data_loader):
        """
        进行验证并返回验证的准确率,和验证loss

        :param data_loader: 验证的dataloader
        :return: 返回验证的准确率
        """
        with torch.no_grad():
            self._net.eval()
            num_correct = 0
            num_total = 0
            for images, labels in data_loader:
                images = images.cuda()
                labels = labels.cuda()

                score = self._net(images)
                loss = self._criterion(score, labels)

                prediction = torch.argmax(score, dim=1)
                num_total += labels.size(0)
                num_correct += torch.sum(prediction == labels).item()
            self._net.train()
        return 100 * num_correct / num_total, loss


def main():
    """
    设置网络的超参数，调用网络进行训练。
    :return: None
    """
    config = configparser.ConfigParser()
    config.read("../../cfg.ini")
    parser = argparse.ArgumentParser(
        description='Train mean field bilinear CNN on Breakhis.'
    )
    parser.add_argument('--Base_lr', dest='base_lr', type=float, default=1e-3,
                        help='Base learning rate for training.')
    parser.add_argument('--Batch_size', dest='batch_size', type=int, default=8,
                        help='batch_size')
    parser.add_argument('--Epochs', dest='epochs', type=int, default=100,
                        help='Epochs for training')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.001,
                        help='Weight decay')
    args = parser.parse_args()

    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must > 0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must > 0')
    if args.epochs <= 0:
        raise AttributeError('--epochs parameter must > 0')
    if args.weight_decay <= 0:
        raise AttributeError('--weight parameter must > 0')

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay
    }

    net = BCNNManager(options=options)
    net.train()


if __name__ == '__main__':
    main()
