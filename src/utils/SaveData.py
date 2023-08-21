"""
    @Author: Panke
    @Time: 2022-11-08  20:53
    @Email: None
    @File: SaveData.py
    @Project: MbsCANet
"""
import os
import json
import time
import shutil
import src.utils.GetVariousPath as path_utils
from src.utils.SaveDataUtils import SaveDataUtils


class SaveData(object):
    @staticmethod
    def save_best_to_file(save_path, cache_path, target_file_name, epoch):
        """
        读取cache文件，将设置的对应的数据保存到指定目录的文件
        :param save_path: 要保存的路径
        :param cache_path: 缓存文件的路径
        :param target_file_name: 目标文件名
        :param epoch: 索引，要保存的哪次epoch的数据
        :return:
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        train_cache = os.path.join(cache_path, 'train_cache.json')
        test_cache = os.path.join(cache_path, 'test_cache.json')

        # 这个Json文件的格式是:{'0':epoch0_dict, '1':epoch1_dict, ...}
        train_data = SaveData.load_json(train_cache)
        # 这个Json文件的格式是:{'error_images':img_num_list, 'total_images':img_num, 'test_acc':test_acc}
        test_data = SaveData.load_json(test_cache)

        best_epoch = epoch  # 这是个整数类型
        train_result = train_data[str(epoch)]  # 这是个dict类型
        error_images = test_data['error_images']  # 这是个list类型
        total_images = test_data['total_images']  # 这里是个整型
        test_acc = test_data['test_acc']  # 这是个浮点数类型

        summary_data = {'best_epoch': best_epoch,
                        'train_result': train_result,
                        'error_images': error_images,
                        'total_images': total_images,
                        'test_acc': test_acc}

        SaveDataUtils.save_dict_to_json(save_path, target_file_name, summary_data)

    @staticmethod
    def save_best_pth(save_path, file_name):
        """
        将pth文件保存至指定目录
        :param save_path: 要保存的目标路径
        :param file_name: pth文件名
        :return: None
        """
        # 获取pth源文件路径
        pth_path = path_utils.get_dir_path('pth')
        # 创建文件源路径
        src_path = os.path.join(pth_path, file_name)
        # 判断目标路径是否存在，如果不存在则创建
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 创建文件目标路径
        target_path = os.path.join(save_path, file_name)

        print(file_name + ' 正在保存 (⊙ˍ⊙)' + '\n\t\t\t\t ... ...')
        # 保存文件
        shutil.copy(src_path, target_path)
        print(file_name + ' 保存完成 (^▽^)')

    @staticmethod
    def load_json(file_path):
        """
        读取一个json文件到一个dict中

        :param file_path: 文件的路径
        :return: 返回一个存储json文件内容的dict
        """
        with open(file_path, 'r', encoding='UTF-8') as file:
            result = json.load(file)
        return result

    @staticmethod
    def save_train_parameters(model_name, magnification, src_list):
        """
        将一个list存储到Excel文件中去

        :param model_name: 模型名称，用来生成文件名和路径有关的东西
        :param magnification: 放大倍数，用来生成文件名和路径有关的东西
        :param src_list: 要保存的list
        :return: None
        """
        # 获取存储Excel文件和Json文件的根路径
        excel_path = path_utils.get_dir_path('excel')
        json_path = path_utils.get_dir_path('json')
        # 获取当前时间
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        # 生成excel文件名
        excel_name = str(model_name) + '_' + str(magnification) + '_' + current_time + '.xlsx'
        # 生成json文件名
        json_name = 'train_cache.json'

        # 生成存Excel文件的分类路径
        excel_path = os.path.join(excel_path, model_name, magnification)

        # 判断文件夹是否存在,如果不存在则创建
        if not os.path.exists(excel_path):
            os.makedirs(excel_path)

        print('训练数据: ' + excel_name + ' 和缓存数据 ' + json_name + ' 正在保存 (⊙ˍ⊙)\n\t\t\t\t... ...')
        # 存储Excel文件和Json文件
        SaveDataUtils.save_list_to_excel(excel_path, excel_name, src_list)
        SaveDataUtils.save_list_to_json(json_path, json_name, src_list)
        print('训练数据: ' + excel_name + ' 和缓存数据 ' + json_name + '保存完成 (^▽^)')

    @staticmethod
    def save_test_parameters(src_dict):
        """
        将测试中的数据保存到缓存文件中

        :param src_dict:
        :return:
        """
        json_path = path_utils.get_dir_path('json')
        json_name = 'test_cache.json'
        print('缓存数据: ' + json_name + ' 正在保存 (⊙ˍ⊙)\n\t\t\t\t... ...')
        SaveDataUtils.save_dict_to_json(json_path, json_name, src_dict)
        print('缓存数据: ' + json_name + ' 保存完成 (^▽^)')
