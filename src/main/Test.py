"""
    @Author: Panke
    @Time: 2022-10-31  19:04
    @Email: None
    @File: Test.py
    @Project: MbsCANet
"""
import os.path

from src.main import Diagnose


# def bind(test):
#     res = {"result": test}
#     return res


if __name__ == '__main__':
    path = "E:/CLFile/MbsCANet/data/200"
    for root, dirs, files in os.walk(path):
        for file in files:
            print(file, end='\t\t')
            print(Diagnose.diagnose(os.path.join(path, file), 200))
