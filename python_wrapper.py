import numpy as np
import matlab_wrapper
import cv2

def imResample(img, hs, ws):
    matlab = matlab_wrapper.MatlabSession()
    matlab.put('img', img)
    matlab.put('hs', hs)
    matlab.put('ws', ws)
    matlab.eval('script')
    im_data = matlab.get('im_data')
    return im_data

def main():
    img = cv2.imread('/home/duino/project/mtcnn/test1.jpg');
    img = img.astype(float)
    hs = 207
    ws = 270
    im_data = imResample(img, hs, ws)
    print im_data.shape
    print im_data[0, 0, :]

if __name__ == "__main__":
    main()
