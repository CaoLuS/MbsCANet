"""
    @Author: Panke
    @Time: 2022-11-08  20:54
    @Email: None
    @File: SaveDataUtils.py
    @Project: MbsCANet
"""
import json
import os
import pandas as pd
import pandas.io.formats.excel

pandas.io.formats.excel.header_style = None


class SaveDataUtils(object):

    @staticmethod
    def save_list_to_excel(excel_path, excel_name, src_list):
        """
        将一个list存储到指定的Excel文件中去。

        :param excel_path 存储文件的路径
        :param excel_name 文件名
        :param src_list: 要存储list对象
        :return: None
        """
        pandas.io.formats.excel.header_style = None
        pf = pd.DataFrame(src_list)
        order = src_list[0].keys()
        pf = pf[order]
        file_path = os.path.join(excel_path, excel_name)
        file_path = pd.ExcelWriter(file_path)
        pf.fillna(' ', inplace=True)
        pf.to_excel(file_path, encoding='utf-8', index=False)
        file_path.save()

    @staticmethod
    def save_list_to_json(json_path, json_name, src_list):
        """
        将一个list存储到指定的json文件（在本项目中可以将所有训练的数据存储在json文件）
        :param json_path: json文件要存储的路径
        :param json_name: json文件的名称
        :param src_list: 要存储list对象
        :return: None
        """
        temp = {}
        # 将列表转换为dict
        for i in range(len(src_list)):
            temp[i + 1] = src_list[i]
        # 将转换后的dict存储到json文件中去
        SaveDataUtils.save_dict_to_json(json_path, json_name, temp)

    @staticmethod
    def save_dict_to_json(dict_path, json_name, src_dict):
        """
        将一个字典存储到一个json文件
        :param dict_path: json文件存储的目录
        :param json_name: json文件名称
        :param src_dict: 要存储的dict对象
        :return:
        """
        file_path = os.path.join(dict_path, json_name)
        json_str = json.dumps(src_dict, indent=4)
        with open(file_path, 'w') as json_file:
            json_file.write(json_str)

    @staticmethod
    def generate_parameter_list(epoch, train_loss, train_acc, test_acc, train_time, parameter_list=None):
        """
        将每个epochs的训练参数保存到一个list中，并返回这个list，返回的list中存储对象是包含每个epoch的字典。
        :param epoch: 迭代次数
        :param train_loss: 训练损失
        :param train_acc: 训练准确率
        :param test_acc: 测试准确率
        :param train_time: 训练时间
        :param parameter_list: 存储每个epoch的list。
        :return: None
        """
        parameter_dict = {'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'test_acc': test_acc,
                          'train_time': train_time}
        parameter_list.append(parameter_dict)
