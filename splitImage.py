"""
splits color image in to 3 images, one for each channel and saves them

Author: Zoe LaLena
Date: 2/7/2023
Course: Senior Project

"""
import cv2 as cv
import sys
import os


def split_and_save(image, dir):
    """
    splits a given image into its three channels and saves each channel to a directory
    :param image: three channel image to split
    :param dir: directory in which images will be saved
    :return: none
    """
    (blue, green, red) = cv.split(image)
    cv.imwrite(dir + "/3blue.jpg", blue)
    cv.imwrite(dir + "/2green.jpg", green)
    cv.imwrite(dir + "/1red.jpg", red)


def combineAndSave(folder):
    image_names = os.listdir(folder)
    blue = cv.imread(folder + "/" + image_names[2],cv.IMREAD_GRAYSCALE)
    red = cv.imread(folder + "/" + image_names[0],cv.IMREAD_GRAYSCALE)
    green = cv.imread(folder + "/" + image_names[1],cv.IMREAD_GRAYSCALE)
    color = cv.merge([red, green, blue])
    cv.imwrite("imageFiles/color_color.png" , color)


def main():
    if len(sys.argv) != 3:
        print("Command Line Arguments: color image location, directory to save channel images")
        return
    path = sys.argv[1]
    image = cv.imread(path)
    split_and_save(image, sys.argv[2])
    #combineAndSave("imageFiles/3 318 r")


if __name__ == '__main__':
    main()
