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

py_faster_rcnn = '/home/duino/project/py-faster-rcnn'

# Add caffe to PYTHONPATH
caffe_path = osp.join(py_faster_rcnn, 'caffe-fast-rcnn', 'python')
add_path(caffe_path)

# Add fast_rcnn to PYTHONPATH
lib_path = osp.join(py_faster_rcnn, 'lib')
add_path(lib_path)
