# MTCNN-v2 face detector (pycaffe version)

Author: zhaoyafei0210@gmail.com (https://github.com/walkoncross)

This repo is based on DuinoDu's MTCNN-v1 (3 stages) repo: https://github.com/DuinoDu/mtcnn, while it implements LNet of MTCNN-v2.

And also 3 wrappers of MTCNN-v2 are included for easy usage.

## Contents
```
├───face_aligner            // mtcnn wrapped as a class 'FaceAligner', which is a super-class of 'MtcnnAligner'. It loads an image and a list of face rects, detects landmarks, does alignment and cropping, outputs aligned face chips.
├───model                   // mtcnn's caffe model
├───mtcnn-duinodu-3stage    // duinodu's 3-stage mtcnn
├───mtcnn_aligner           // mtcnn warpped as a class 'MtcnnAligner' with 3 stages (RNet, ONet, LNet), PNet removed. It loads an image and a list of face rects, and detects 5 landmarks for each rects, output a list of 5 landmarks.
├───mtcnn_detector          // mtcnn warpped as a class 'MtcnnDetector' with 4 stages (PNet, RNet, ONet, LNet).
├───scripts                 // some self-used test scripts
├───test_imgs               // test images
```
