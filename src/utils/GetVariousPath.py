"""
    @Author: Panke
    @Time: 2022-11-08  20:51
    @Email: None
    @File: GetVariousPath.py
    @Project: MbsCANet
"""
import os


def get_project_path():
    """
    获取项目的根路径, 这个文件必须放在：项目根目录/src/utils下

    :return: 返回项目的根路径
    """
    return os.path.abspath(os.path.join(os.getcwd(), '../..'))


def get_data_dir_path(data_dir):
    """
    获取存放数据集的文件夹的路径，文件夹位于项目根目录之下，文件名夹名称是data_dir

    :param data_dir: 位于项目根路径下的存放数据集的文件夹的名称
    :return: 返回存放数据集的文件夹路径
    """
    project_path = get_project_path()
    return os.path.join(project_path, data_dir)


def get_dataset_dir_path(data_dir, dataset_name):
    """
    获取数据集的路径

    :param data_dir: 存放数据集的文件夹的名称。
    :param dataset_name: 数据集的名称(数据集的文件夹名称)
    :return: 返回数据集的路径
    """
    data_dir = get_data_dir_path(data_dir)
    return os.path.join(data_dir, dataset_name)


def get_dataset_magnification_path(data_dir, dataset_dir, magnification_dir):
    """
    获取数据集中相关倍数的数据集

    :param data_dir: 存放数据的文件夹名称，例如：项目根目录下的data文件夹。
    :param dataset_dir: 数据集的路径，例如：data文件夹下的Breakhis文件夹。
    :param magnification_dir: 数据放大倍数的文件夹，有40x、100x、200x、400x，例如：Breakhis文件夹下有各个放大倍数的文件夹。
    :return: 返回一个数据集中倍率为magnification_dir的文件夹。
    """
    data_dir = get_data_dir_path(data_dir)
    dataset_dir = get_dataset_dir_path(data_dir, dataset_dir)
    return os.path.join(data_dir, dataset_dir, magnification_dir)


def get_dir_path(dir_name):
    """
    获取项目根目录下名为dir_name文件夹的绝对路径

    :param dir_name: 要获取的文件夹的名称
    :return: 返回dir_name的绝对路径。
    """
    project_path = get_project_path()
    return os.path.join(project_path, dir_name)


def get_save_pth_file(pth_dir, pth_file_name):
    """
    获取要存储的pth文件

    :param pth_dir: pth文件的存储目录
    :param pth_file_name: pth文件的文件名
    :return: 返回要存储的pth文件
    """
    pth_dir = get_dir_path(pth_dir)
    return os.path.join(pth_dir, pth_file_name)
