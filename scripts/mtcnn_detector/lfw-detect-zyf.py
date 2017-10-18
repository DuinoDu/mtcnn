#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import os.path as osp

import cv2
# import numpy as np

import time
import json

import _init_paths
from mtcnn_detector import MtcnnDetector, draw_faces


def print_usage():
    usage = 'python %s <img-list-file> <save-dir>' % osp.basename(__file__)
    print('USAGE: ' + usage)


def main(lfw_list_fn,
         lfw_root,
         save_dir,
         save_img=False,
         show_img=False):

    minsize = 20
    caffe_model_path = "../../model"
    threshold = [0.6, 0.7, 0.7]
    scale_factor = 0.709

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fp_rlt = open(osp.join(save_dir, 'lfw_mtcnn_fd_rlt.json'), 'w')

#    result_list = []
    fp_rlt.write('[\n')

    t1 = time.clock()
    detector = MtcnnDetector(caffe_model_path)
    t2 = time.clock()
    print("initFaceDetector() costs %f seconds" % (t2 - t1))

    fp = open(lfw_list_fn, 'r')

    ttl_time = 0.0
    img_cnt = 0

    for line in fp:
        imgpath = line.strip()
        print("\n===>" + imgpath)
        if imgpath == '':
            print 'empty line, not a file name, skip to next'
            continue
        if imgpath[0] == '#':
            print 'skip line starts with #, skip to next'
            continue

        splits = imgpath.split()
        imgpath = splits[0]

        id = 'unkown' if len(splits) < 2 else splits[1]

        if not imgpath.startswith('/'):
            fullpath = osp.join(lfw_root, imgpath)
        else:
            fullpath = imgpath

        rlt = {}
        rlt["filename"] = imgpath
        rlt["faces"] = []
        rlt['face_count'] = 0
        rlt['id'] = id

        try:
            img = cv2.imread(fullpath)
        except:
            print('failed to load image: ' + fullpath)
            rlt["message"] = "failed to load"
            result_list.append(rlt)
            continue

        if img is None:
            print('failed to load image: ' + fullpath)
            rlt["message"] = "failed to load"
            result_list.append(rlt)
            continue

        img_cnt += 1
        t1 = time.clock()

        bboxes, points = detector.detect_face(img, minsize,
                                              threshold, scale_factor)

        t2 = time.clock()
        ttl_time += t2 - t1
        print("detect_face() costs %f seconds" % (t2 - t1))

        if len(bboxes) > 0:
            for (box, pts) in zip(bboxes, points):
                #                box = box.tolist()
                #                pts = pts.tolist()
                tmp = {'rect': box[0:4],
                       'score': box[4],
                       'pts': pts
                       }
                rlt['faces'].append(tmp)

            rlt['face_count'] = len(bboxes)
        rlt['message'] = 'success'
#        result_list.append(rlt)
        s = json.dumps(rlt, indent=2)
        fp_rlt.write(s + ',\n')
#        fp_rlt.write(',\n' + s)

#        print('output bboxes: ' + str(bboxes))
#        print('output points: ' + str(points))
        # toc()

        if bboxes is None:
            continue

        print("\n===> Processed %d images, costs %f seconds, avg time: %f seconds" % (
            img_cnt, ttl_time, ttl_time / img_cnt))

        if save_img or show_img:
            draw_faces(img, bboxes, points)

        if save_img:
            save_name = osp.join(save_dir, osp.basename(imgpath))
            cv2.imwrite(save_name, img)

        if show_img:
            cv2.imshow('img', img)

            ch = cv2.waitKey(0) & 0xFF
            if ch == 27:
                break

#    json.dump(result_list, fp_rlt, indent=2)
#    print fp_rlt.tell()

    # delete the last ','
    if sys.platform is 'win32':
        fp_rlt.seek(-3, 1)
    else:
        fp_rlt.seek(-2, 1)
    fp_rlt.write('\n]')

    fp_rlt.close()
    fp.close()

    if show_img:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print_usage()

#    lfw_list_fn = "./lfw_list_part.txt"
    lfw_list_fn = "./list_lfw_failed3.txt"
#    lfw_list_fn = "lfw_list_mtcnn.txt"
    save_dir = './lfw_rlt'
#    lfw_root = '/disk2/data/FACE/LFW/LFW'
    lfw_root = r'C:\zyf\dataset\lfw'

    print(sys.argv)

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    if len(sys.argv) > 1:
        lfw_list_fn = sys.argv[1]

    if len(sys.argv) > 2:
        save_dir = sys.argv[2]

    if len(sys.argv) > 3:
        show_img = not(not(sys.argv[3]))

#    main(lfw_list_fn, lfw_root, save_dir, save_img=True, show_img=True)
    main(lfw_list_fn, lfw_root, save_dir, save_img=False, show_img=False)
