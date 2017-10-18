# --------------------------------------------------------
# xxx
# Copyright (c) 2016 xxx
# Licensed under The MIT License [see LICENSE for details]
# Written by Duino Du
# --------------------------------------------------------

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.abspath('../../mtcnn_detector'))
