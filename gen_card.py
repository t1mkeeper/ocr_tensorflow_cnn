#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
读取身份证照片，并做些处理，看看效果

@author: pengyuanjie
"""
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  
import math
  
fn = "card2.jpg"

myimg = cv2.imread(fn)

#gray = myimg[:,:,2]

gray = cv2.cvtColor(myimg,cv2.COLOR_BGR2GRAY) 
#gradX = cv2.Sobel(gray, ddepth=cv2.cv.CV_32F, dx=1, dy=0, ksize=-1)
#gradY = cv2.Sobel(gray, ddepth=cv2.cv.CV_32F, dx=0, dy=1, ksize=-1)

# subtract the y-gradient from the x-gradient
#gradient = cv2.subtract(gradX, gradY)
#gradient = cv2.convertScaleAbs(gradient)


# blur and threshold the image
#双边滤波
#gray = cv2.bilateralFilter(gray,9,75,75)
#均值滤波
#gray = cv2.blur(gray, (20, 20))
#中值滤波
#gray = cv2.medianBlur(gray, 3)
#高斯模糊滤波
#blurred = cv2.GaussianBlur(gray,(5,5),1.5)

#光照矫正

cv2.imshow('gray_old',gray)

arr = np.asarray(gray)
new_arr = arr.copy()
(x,y) = arr.shape

size = 50
x_ = int(math.ceil(x / (size * 1.0)))
y_ = int(math.ceil(y / (size * 1.0)))

for i in range(0, x_):
    for j in range(0, y_):
        sub_arr = gray[i*size:(i+1)*size,j*size:(j+1)*size]
        #计算区域均值
        avg = np.mean(sub_arr)
        #计算区域标准差
        std = np.std(sub_arr, ddof = 1)
        sub_val = max(np.min(sub_arr),avg - 3*std)
        #sub_val = np.min(sub_arr)
        for m in range(i*size, (i+1)*size):
            if m >= x:
                break
            for n in range(j*size, (j+1)*size):
                if n >= y:
                    break
                new_arr[m][n] = sub_val
                gray[m][n] =  max(0,gray[m][n]-sub_val)
                gray[m][n] =  min(255,gray[m][n] + 50)

cv2.imshow('bak',new_arr)

gray = cv2.medianBlur(gray, 5)
cv2.imshow('gray_new',gray)
T = 120
#计算最佳阈值
while True:
    smax = 0.0
    cmax = 0
    smin = 0.0
    cmin = 0
    for i in range(0,x):
        for j in range(0,y):
            if (arr[i][j] >= T):
                smax += arr[i][j]
                cmax +=1
            else:
                smin += arr[i][j]
                cmin += 1
    if cmax == 0:
        u1 = 0
    else:
        u1 = smax / cmax
    if cmin == 0:
        u2 = 0
    else:
        u2 = smin / cmin
    T_new = (u1 + u2) / 2
    print T_new
    if math.floor(abs(T - T_new)) == 0:
        T = T_new
        break
    T = T_new
    
(_, thresh) = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)

#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# perform a series of erosions and dilations
#thresh = cv2.erode(thresh, None, iterations=1)
#thresh = cv2.dilate(thresh, None, iterations=1)

cv2.imshow('thresh',thresh)
'''
cv2.findContours()函数,第二个参数为检测模式
cv2.RETR_EXTERNAL表示只检测外轮廓
cv2.RETR_LIST检测的轮廓不建立等级关系
cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
cv2.RETR_TREE建立一个等级树结构的轮廓。
'''
(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
c_all = sorted(cnts, key=cv2.contourArea, reverse=True)

l = len(c_all)
for i in range(0,0):
    c = c_all[i]
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.cv.BoxPoints(rect))
    
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    cropImg = myimg[y1:y1+hight, x1:x1+width]
    cv2.imwrite("result_"+ str(i) +".jpg", cropImg)

# draw a bounding box arounded the detected barcode and display the image
cv2.drawContours(myimg, c_all, -1, (0, 255, 0), 3)
cv2.imshow('myimg',myimg)

cv2.waitKey(0) 
cv2.destroyAllWindows() 

  

