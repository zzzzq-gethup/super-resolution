# flake8: noqa
import os
import os.path as osp
import sys


# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# 将上级目录添加到 PYTHONPATH
sys.path.append(parent_dir)

from fscsr.archs import *
from fscsr.data import *
from fscsr.models import *

from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
