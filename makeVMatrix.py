"""

"""
import cv2 as cv
import numpy as np
import os


def getV(image, save=False, fileName=""):
    w, h = image.shape
    roi_list = []
    scale_percent = 500  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    view_able = cv.resize(image,dim, interpolation=cv.INTER_AREA)
    while cv.waitKey() != 27:
        region = cv.selectROI("Select Class Pixels", view_able)
        cv.moveWindow("Select Class Pixels", 40, 30)
        scaled = []
        scaled.append(region[0] / (scale_percent/100))
        scaled.append(region[1] / (scale_percent/100))
        scaled.append(region[2] / (scale_percent/100))
        scaled.append(region[3] / (scale_percent/100))
        roi_list.append(scaled)

    # make v have width and height equal to number of pixels in image
    v_size = w * h
    V = np.zeros((v_size, v_size))
    for roi in roi_list:
        for r in range(round(roi[0]), round(roi[0] + roi[2])):
            for c in range(round(roi[1]), round(roi[1] + roi[3])):
                # this should give the 1d location of the pixel we are looking at
                location = r * w + c + 1
                V[location, location] = 1
    if save:
        file = open("testFiles/" + fileName, "w")
        for r in range(0, v_size):
            for c in range(0, v_size):
                file.write(str(V[r, c]))
                if c != v_size - 1:
                    file.write(",")
            file.write("\n")
        file.close()
    return V


def crop(folder):
    image_names = os.listdir(folder)
    image1 = cv.imread(folder + "/" + image_names[5],cv.IMREAD_GRAYSCALE)

    w, h = image1.shape
    roi_list = []
    scale_percent = 200  # percent of original size
    width = int(image1.shape[1] * scale_percent / 100)
    height = int(image1.shape[0] * scale_percent / 100)
    dim = (width, height)
    view_able = cv.resize(image1, dim, interpolation=cv.INTER_AREA)
    region = cv.selectROI("Crop", view_able)
    cv.moveWindow("Crop", 40, 30)
    scaled = []
    scaled.append(region[0] / (scale_percent / 100))
    scaled.append(region[1] / (scale_percent / 100))
    scaled.append(region[2] / (scale_percent / 100))
    scaled.append(region[3] / (scale_percent / 100))
    roi_list.append(scaled)


    for img_name in image_names:
        img = cv.imread(folder + "/" + img_name,cv.IMREAD_GRAYSCALE)
        img_crop = img[int(scaled[1]):int(scaled[1] + scaled[3]), int(scaled[0]):int(scaled[0] + scaled[2])]
        cv.imwrite("imageFiles/318_all_aspects_half/" + img_name, img_crop)


def main():
    crop("imageFiles/318_all_aspects")
    # image = cv.imread("imageFiles/cropped 318r/1crop_ msXL_318r_b-M0627RD_12_F.tif",cv.IMREAD_GRAYSCALE)
    # getV(image, True, "318_cropped.csv")


if __name__ == '__main__':
    main()
