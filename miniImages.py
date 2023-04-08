"""
Code to make a bunch of smaller images from a larger one
File: miniImages.py
Author: Zoe LaLena
Date: 4/4/2023
Course: Senior Project
"""
import os

import DWkNNFromROI
import cv2 as cv
import sys

def splitImage(folder, save_location):
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    image_names = os.listdir(folder)
    images = DWkNNFromROI.make_image_list(folder)
    # want images to be about 100x100
    w, h = images[0].shape
    divisor = 100
    # let figure out how many images fir in each dimension
    w_amount = int(w / 100)
    h_amount = int(h / 100)
    img_count = 0
    total_images = w_amount*h_amount
    img_count = 0
    for r in range(0, w_amount):
        for c in range(0, h_amount):
            newpath = save_location +"/" + str(divisor*r) +"_" + str(divisor*c)
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            cur_img = 0
            for img in images:
                section = img[divisor*r:divisor*(r+1),divisor*c:divisor*(c+1)]
                cv.imwrite(newpath+"/" + image_names[cur_img], section)
                cur_img = cur_img+1


def main():
    if len(sys.argv) != 3:
        print("Command Line Arguments: path to folder of images, save location")
        return
    else:
        splitImage(sys.argv[1],sys.argv[2])

if __name__ == '__main__':
    main()