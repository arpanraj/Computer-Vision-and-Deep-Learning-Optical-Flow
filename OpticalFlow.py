#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 23:32:50 2022

@author: arpanrajpurohit
"""

#load Images
import numpy as np
from PIL import Image
import math
import cv2 as cv
import matplotlib.pyplot as plt

FOLDER_PATH = "/Users/arpanrajpurohit/Desktop/courses/Computer_Vision_and_Deep_Learning_510/Programs/Programming2/"
INPUT_PATH  = FOLDER_PATH + "input/"
OUTPUT_PATH = FOLDER_PATH + "output/"

#IMAGE PRE PROCESSING
IMAGE_NAMES = ["frame1_a.png", "frame1_b.png", "frame2_a.png", "frame2_b.png"]
GRAYSCALE_MODE = 'L'
PAD_MODE = 'constant'
padded_image_arrs = []
image_arrs = []
gray_image_arrs = []
for image_name in IMAGE_NAMES:
    rgb_image = cv.imread(INPUT_PATH + image_name)
    gray_image = cv.cvtColor(rgb_image, cv.COLOR_BGR2GRAY)
    image_arrs.append(rgb_image)
    gray_image_arrs.append(gray_image)
    paded_img_arr = np.pad(gray_image, 1, mode=PAD_MODE)
    padded_image_arrs.append(paded_img_arr)

video_arrs = [[0, 1], [2, 3]]


# Derivative of Gaussian Filter X

DOGx_arr_mul       = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
def DOGx_filter(pade1_image_arr, row, col):
    ans = np.sum(DOGx_arr_mul * pade1_image_arr[row: row + DOGx_arr_mul.shape[0], col: col + DOGx_arr_mul.shape[1]])
    return ans

# Derivative of Gaussian Filter Y

DOGy_arr_mul        = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
def DOGy_filter(pade1_image_arr, row, col):
    ans = np.sum(DOGy_arr_mul * pade1_image_arr[row: row + DOGy_arr_mul.shape[0], col: col + DOGy_arr_mul.shape[1]])
    return ans

# Intesity Difference in frames

frame1_int_mul        = np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]])
frame2_int_mul        = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

def Intesity_dif(frame1, frame2, row, col):
    ans = np.sum((frame1_int_mul * frame1[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]])) + np.sum((frame2_int_mul * frame2[row: row + frame2_int_mul.shape[0], col: col + frame2_int_mul.shape[1]]))
    return ans


#video processing
for video in video_arrs:
    image1 = image_arrs[video[0]]
    image2 = image_arrs[video[1]]
    gray_image1 = gray_image_arrs[video[0]]
    gray_image2 = gray_image_arrs[video[1]]
    frame1 = padded_image_arrs[video[0]]
    frame2 = padded_image_arrs[video[1]]
    dogx = np.empty(gray_image1.shape)
    dogy = np.empty(gray_image1.shape)
    intesity_dif = np.empty(gray_image1.shape)
    
    for row in range(image1.shape[0]):
        for col in range(image1.shape[1]):
            dogx[row][col] = DOGx_filter(frame1, row, col)
            dogy[row][col] = DOGy_filter(frame1, row, col)
            intesity_dif[row][col]  = Intesity_dif(frame1, frame2, row, col)
    # vx = np.empty(image1.shape)
    # vy = np.empty(image1.shape)
    # A = np.zeros((2, 2))
    # B = np.zeros((2, 1))
    # for row in range(image1.shape[0]):
    #     for col in range(image1.shape[1]):
    #         A[0, 0] = np.sum((dogx[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]]) ** 2)
    #         A[0, 1] = np.sum((dogx[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]]) * (dogy[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]]))
    #         A[1, 0] = np.sum((dogx[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]]) * (dogy[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]]))
    #         A[1, 1] = np.sum((dogy[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]]) ** 2)
    #         Ainv = np.linalg.pinv(A)
    #         B[0, 0] = -np.sum(dogx[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]] * intesity_dif[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]])
    #         B[1, 0] = -np.sum(dogy[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]] * intesity_dif[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]])
    #         prod = np.matmul(Ainv, B)
    #         vx[row][col] = prod[0]
    #         vy[row][col] = prod[1]
    # sqrtofvxvy = np.sqrt((vx ** 2) + (vy ** 2))
    # parameter to get features
    params = dict(maxCorners=2000,
                          qualityLevel=0.001,
                          minDistance=4,
                          blockSize=4)
    
    track= cv.goodFeaturesToTrack(gray_image1, mask = None, **params)  #using opencv function to get feature for which we are plotting flow
    trackers = np.int32(track)
    # print(feature)
    tracker = np.reshape(trackers, newshape=[-1, 2])
    
    vx = np.ones(dogx.shape)
    vy = np.ones(dogx.shape)
    status=np.zeros(tracker.shape[0]) # this will tell change in x,y
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))
    mask = np.zeros_like(image1)
    newTracker=np.zeros_like(tracker)
    
    for a,i in enumerate(tracker):
    
        x, y = i
    
        A[0, 0] = np.sum((dogx[y - 1:y + 2, x - 1:x + 2]) ** 2)
    
        A[1, 1] = np.sum((dogy[y - 1:y + 2, x - 1:x + 2]) ** 2)
        A[0, 1] = np.sum(dogx[y - 1:y + 2, x - 1:x + 2] * dogy[y - 1:y + 2, x - 1:x + 2])
        A[1, 0] = np.sum(dogx[y - 1:y + 2, x - 1:x + 2] * dogy[y - 1:y + 2, x - 1:x + 2])
        Ainv = np.linalg.pinv(A)
    
        B[0, 0] = -np.sum(dogx[y - 1:y + 2, x - 1:x + 2] * intesity_dif[y - 1:y + 2, x - 1:x + 2])
        B[1, 0] = -np.sum(dogy[y - 1:y + 2, x - 1:x + 2] * intesity_dif[y - 1:y + 2, x - 1:x + 2])
        prod = np.matmul(Ainv, B)
    
        vx[y, x] = prod[0]
        vy[y, x] = prod[1]
    
        newTracker[a]=[np.int32(x+vx[y,x]),np.int32(y+vy[y,x])]
        if np.int32(x+vx[y,x])==x and np.int32(y+vy[y,x])==y:    # this means that there is no change(x+dx==x,y+dy==y) so marking it as 0 else
            status[a]=0
        else:
            status[a]=1 # this tells us that x+dx , y+dy is not equal to x and y
    
    um=np.flipud(vx)
    vm=np.flipud(vy)
    good_new=newTracker[status==1] #status will tell the position where x and y are changed so for plotting getting only that points
    good_old = tracker[status==1]
    # draw the tracks
    for ind, (new, old) in enumerate(zip(good_new, good_old)):
        q, w = new.ravel()
        a, s = old.ravel()
        mask = cv.line(mask, (q, w), (a, s), (0,255,0), 2)
        newframe = cv.circle(image2, (q, w), 1, (0,255,0), -1)
    image = cv.add(newframe, mask)
    cv.imwrite(OUTPUT_PATH + "optical_flow_generated" + IMAGE_NAMES[video[0]], image)
    plt.imshow(image)
    plt.show()
    vx = np.empty(gray_image1.shape)
    vy = np.empty(gray_image1.shape)
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))
    for row in range(image1.shape[0]):
        for col in range(image1.shape[1]):
            A[0, 0] = np.sum((dogx[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]]) ** 2)
            A[0, 1] = np.sum((dogx[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]]) * (dogy[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]]))
            A[1, 0] = np.sum((dogx[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]]) * (dogy[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]]))
            A[1, 1] = np.sum((dogy[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]]) ** 2)
            Ainv = np.linalg.pinv(A)
            B[0, 0] = -np.sum(dogx[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]] * intesity_dif[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]])
            B[1, 0] = -np.sum(dogy[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]] * intesity_dif[row: row + frame1_int_mul.shape[0], col: col + frame1_int_mul.shape[1]])
            prod = np.matmul(Ainv, B)
            vx[row][col] = prod[0]
            vy[row][col] = prod[1]
    sqrtofvxvy = np.sqrt((vx ** 2) + (vy ** 2))
    filtered_img = Image.fromarray(sqrtofvxvy)
    filtered_img.show()
           