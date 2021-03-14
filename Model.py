import os
import re
import cv2
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

col_frames = os.listdir('frames/')

col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

col_images=[]

for i in col_frames:

    img = cv2.imread('frames/'+i)

    col_images.append(img)


kernel = np.ones((4,4),np.uint8)

pathIn = "contour_frames_3/"

for i in range(len(col_images)-1):

    grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(col_images[i+1], cv2.COLOR_BGR2GRAY)
    diff_image = cv2.absdiff(grayB, grayA)

    ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

    dilated = cv2.dilate(thresh,kernel,iterations = 1)

    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    valid_cntrs = []
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        if (x <= 200) & (y >= 80) & (cv2.contourArea(cntr) >= 25):
            if (y >= 90) & (cv2.contourArea(cntr) < 40):
                break
            valid_cntrs.append(cntr)
    dmy = col_images[i].copy()
    cv2.drawContours(dmy, valid_cntrs, -1, (127,200,0), 2)
    
    cv2.putText(dmy, "vehicles detected: " + str(len(valid_cntrs)), (55, 15), 0.6, (0, 180, 0), 2)
    cv2.line(dmy, (0, 80),(256,80),(100, 255, 255))
    cv2.imwrite(pathIn+str(i)+'.png',dmy)

pathOut = 'vehicle_detection_v3.mp4'

fps = 14.0

frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

files.sort(key=lambda f: int(re.sub('\D', '', f)))

for i in range(len(files)):

    filename=pathIn + files[i]
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    frame_array.append(img)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frame_array)):
    out.write(frame_array[i])

out.release()