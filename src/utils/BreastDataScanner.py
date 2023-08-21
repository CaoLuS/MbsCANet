"""
    @Author: Panke
    @Time: 2022-11-08  20:38
    @Email: None
    @File: BreastDataScanner.py
    @Project: MbsCANet
"""
import os

import PIL.Image
import skimage.io as io
from torch.utils import data


class DataScanner(data.Dataset):
    """
    加载并处理各个数据集
    """

    def __init__(self, data_path, for_what, transform=None, target_transform=None, download=False):
        """
        初始化成员变量

        :param data_path: 要加载的数据的路径（在本例中也就是放大倍数倍数的那个文件夹）
        :param for_what: 加载哪些数据，也就是用来干什么，可选的有train、validate和test，分表表示加载训练数据集，验证数据集和测试数据集
        :param transform: transform数据处理
        :param target_transform: 标签数据处理
        :param download: 是否下载
        """
        self.data_path = data_path
        self.for_what = for_what
        self._transform = transform

        if self.for_what == 'train':
            self.data = os.path.join(self.data_path, 'train', '1')
            name_path = os.path.join(self.data_path, 'train', 'label', '1zhe_train.txt')
        elif self.for_what == 'validate':
            self.data = os.path.join(self.data_path, 'train', '1')
            name_path = os.path.join(self.data_path, 'train', 'label', '1zhe_val.txt')
        elif self.for_what == 'test':
            self.data = os.path.join(self.data_path, 'test')
            name_path = os.path.join(self.data_path, 'test', 'test_label.txt')
        else:
            raise TypeError('传入的 for_what 类型错误，for_what 必须为 train 或者 validate 或者 test')

        # 逐行读取标签文件，把文件的路径和标签添加到images列表中
        self.images = read_label_file(name_path, self.data)

    def __getitem__(self, index):
        image, label = read_image(self.images, index, self._transform)
        return image, label

    def __len__(self):
        """
        返回数据集的长度
        :return: 返回数据集的长度
        """
        return len(self.images)


def read_label_file(file_path, data_path):
    """
    读取数据标签文件，返回一个存储每个数据对应标签的列表

    :param file_path: 标签文件的路径
    :param data_path: 数据的路径
    :return: 返回一个存储每个数据对应标签的列表
    """
    result = []
    file = open(file_path, 'r')
    for line in file:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        result.append((os.path.join(data_path, words[0]), int(words[1])))
    return result


def read_image(images, index, transform=None):
    """
    读取并处理图片，返回处理后的图片和标签

    :param images: 存储有图片路径和其标签的列表
    :param index: 索引
    :param transform: 数据处理的transform
    :return: 返回处理后的图片和标签
    """
    image, label = images[index]
    image = io.imread(image)
    image = PIL.Image.fromarray(image)
    if transform is not None:
        image = transform(image)
    return image, label



