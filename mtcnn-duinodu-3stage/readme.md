# mtcnn

It a python version of [mtcnn](https://github.com/kpzhang93/MTCNN_face_detection_alignment), which is a face detection using cnn.

### Requirement
0. ubuntu
1. caffe && pycaffe: [https://github.com/BVLC/caffe](https://github.com/BVLC/caffe), [my csdn](http://blog.csdn.net/duinodu/article/details/52760587)
2. opencv && cv2: [my csdn](http://blog.csdn.net/duinodu/article/details/51804642)
3. numpy

### Tell mtcnn where pycaffe is
Edit mtcnn/_init_paths.py, change **caffe_path** to your own. 

### Run
```
git clone https://github.com/DuinoDu/mtcnn && cd mtcnn
./run
```
