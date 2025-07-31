# flake8: noqa
import os
import os.path as osp
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# 将上级目录添加到 PYTHONPATH
sys.path.append(parent_dir)

import fscsr.archs
import fscsr.data
import fscsr.models
from basicsr.test import test_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
