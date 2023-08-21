"""
    @Author: Panke
    @Time: 2022-11-08  23:36
    @Email: None
    @File: Diagnose.py
    @Project: MbsCANet
"""
import os.path

import torch
from PIL import Image
from torchvision.transforms import transforms
import src.utils.GetVariousPath as path_utils
from src.models.MbsCANet import MbsCANet
import src.utils.SaveData as SaveData
from src.main.Heatmap import heatmap


def diagnose(img, magnification):
    """
    输入给定的图片，识别图片属于哪个分类
    :param img: 图像的地址
    :param magnification: 图像的放大倍数
    :return:
    """
    class_names = ['0', '1']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                    transforms.ToTensor(), ])

    weight_file_path = path_utils.get_dir_path("weights")
    weight_file_path = os.path.join(weight_file_path, str(magnification), "weight.pth")
    model = torch.nn.DataParallel(MbsCANet())

    # 加载训练模型
    model.load_state_dict(torch.load(weight_file_path))
    # torch.nn.
    model.to(device)
    model.eval()

    heat = heatmap(magnification, img)

    # 加载图片
    img = Image.open(img)
    img_ = transform(img)
    img_ = torch.unsqueeze(img_, dim=0)
    img_ = img_.to(device)

    # 将图片放入网络
    outputs = model(img_)
    _, index = torch.max(outputs, 1)

    # 获取输出结果最大可能的分类的百分比
    percentage = torch.nn.functional.softmax(outputs, dim=1)[0]
    percentage = percentage.cpu().detach().numpy() * 100

    # 获取输出的结果中最大可能的分类
    result_perc = round(percentage[int(index)], 2)
    # result_perc = percentage[int(index)]
    result_class = class_names[int(index)]

    # 将分类和可能的百分比放入字典中
    result_dict = dict()
    result_dict['heatmap'] = heat
    result_dict['class'] = result_class
    result_dict['percentage'] = round(float(result_perc), 3)
    # 保存数据到json文件
    SaveData.SaveDataUtils.save_dict_to_json(path_utils.get_dir_path('json'), 'result_json.json', result_dict)
    return result_dict
