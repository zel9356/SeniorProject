"""
Graces data is inputted as [Pixel col, Pixel row, intensity in N Bands
excluding the row and colm info in the first two cols, the data is an MxN CSV file where M is the number of pixels
and N is the number of bands

Author: Zoe LaLena
Course: Senior Project

Arguments:
"""
# images: C:\Users\zobok\School\Semster 7 (Fall 2022)\Senior Project\Code\kNN\Manu Images
import os
import cv2 as cv
import numpy as np

def grabROI(image):
    cv.namedWindow("Resized_Window", cv.WINDOW_NORMAL)

    scale_percent = 50  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    roi = cv.selectROI("Resized_Window", resized)

    # now we need to rescale
    scaled = []
    scaled.append(roi[0]/.5)
    scaled.append(roi[1] / .5)
    scaled.append(roi[2] / .5)
    scaled.append(roi[3] / .5)
    return scaled

def makeCSVFile(images, roi):
    """
    Makes a CSV file of pixel row and colm in the first twi columns of the file, followed by the intensity in each image
    in the following cols
    :param images: list of images to be added to the csv file
    :return:
    """
    file = open("../../Downloads/filename.csv", "w")
    w, h = (images[0]).shape
    print(w)
    # first write row and colm
    for row in range(int(roi[1]), int(roi[1]+roi[3])):
        for col in range(int(roi[0]),int(roi[0]+roi[2])):
            for img in range(0, len(images)):
                if img == 0:
                    file.write(str(col) + "," + str(row))
                file.write(",")
                file.write(str(images[img][row, col]/65536))
            file.write("\n")
    file.close()

#I think the easiest way to do this is open all the images and put them in a list
def makeImageList(path):
    """
    Makes a list of all the images in a given directory
    :param path:  containing image files
    :return: list of images
    """
    image_names = os.listdir(path)
    print(image_names)
    images = []
    for image in image_names:
        img = cv.imread(path+ "\\" + image,  cv.COLOR_BGR2GRAY)
        images.append(img)
    return images

def main():
    imageList = makeImageList(r"D:\Zoe\318r\TIFFs")
    image1 = imageList[9]
    roiCords = grabROI(image1)
    print(roiCords)
    makeCSVFile(imageList, roiCords)

if __name__ == '__main__':
    main()
