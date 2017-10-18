# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 06:33:36 2017

@author: zhaoy
"""
import sys
import os
import os.path as osp

import json
import time
import cv2

import _init_paths
from mtcnn_aligner import MtcnnAligner, draw_faces


def main(json_file, save_dir=None, save_img=True, show_img=True):
    if not osp.exists(json_file):
        print 'Cannot find json file: ' + json_file
        pass

    if save_dir is None:
        save_dir = './fa_facex_rlt'

    save_json = 'mtcnn_align_rlt.json'
    caffe_model_path = "../../model"

    fp_json = open(json_file, 'r')
    facex_response = json.load(fp_json)
    fp_json.close()

    if (not facex_response
            or not isinstance(facex_response, dict)
            or 'facex_det' not in facex_response
            ):
        print 'Invalid json file: ' + json_file
        pass

    facex_det_response = facex_response['facex_det']

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fp_rlt = open(osp.join(save_dir, save_json), 'w')
    results = []

    for item in facex_det_response:
        img_path = item['name']
        print '===> Processing image: ' + img_path

        if 'detections' not in item:
            continue

        face_rects = []
        for face in item['detections']:
            face_rects.append(face['pts'])

        img = cv2.imread(img_path)

        aligner = MtcnnAligner(caffe_model_path, False)

        rlt = {}
        rlt["filename"] = img_path
        rlt["faces"] = []
        rlt['face_count'] = 0

        t1 = time.clock()
        bboxes, points = aligner.align_face(img, face_rects)
        t2 = time.clock()

        n_boxes = len(face_rects)
        print("-->Alignment cost %f seconds, processed %d face rects, avg time: %f seconds" %
              ((t2 - t1), n_boxes, (t2 - t1) / n_boxes))

        if bboxes is not None and len(bboxes) > 0:
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
        results.append(rlt)

        if save_img or show_img:
            draw_faces(img, bboxes, points)

        if save_img:
            pdir = save_dir
            dir_name, base_name = osp.split(img_path)
            if dir_name:
                _, pdir = osp.split(dir_name)
                if pdir:
                    pdir = osp.join(save_dir, pdir)
                    if not osp.exists(pdir):
                        os.makedirs(pdir)

            save_name = osp.join(pdir, osp.basename(img_path))
            cv2.imwrite(save_name, img)

        if show_img:
            cv2.imshow('img', img)
            cv2.waitKey(0)

            cv2.destroyAllWindows()

    json.dump(results, fp_rlt, indent=4)
    fp_rlt.close()

if __name__ == '__main__':
#    json_file = './facex_det_response.json'
    json_file = r'C:\zyf\github\misc-zyf\facex_det_wlc_cvt.json'
    save_dir = None

    main(json_file, save_dir, save_img=True, show_img=False)
