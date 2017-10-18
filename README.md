# MTCNN face detector (pycaffe version)

Author: zhaoyafei0210@gmail.com (https://github.com/walkncross)

This repo is based on DuinoDu's repo: https://github.com/DuinoDu/mtcnn

## Contents
```
├───face_aligner            // mtcnn wrapped as a class 'FaceAligner', which is a super-class of 'MtcnnAligner'. It reads a image and a list of face rects, detects landmarks, does alignment and cropping, outputs aligned face chips.
├───model                   // mtcnn's caffe model
├───mtcnn-duinodu-3stage    // duinodu's 3-stage mtcnn
├───mtcnn_aligner           // mtcnn warpped as a class 'MtcnnAligner' with 3 stages (RNet, ONet, LNet), PNet removed. It reads a image and a list of face rects, and detects 5 landmarks for each rects, output a list of landmarks.
├───mtcnn_detector          // mtcnn warpped as a class 'MtcnnDetector' with 4 stages (PNet, RNet, ONet, LNet).
├───test_imgs               // test images
```
