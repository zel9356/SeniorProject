"""
code to handle pre-selected V values
File: presetV.py
Author: Zoe LaLena
Date: 3/23/2023
Course: Senior Project
"""
import cv2 as cv
import DWkNNFromROI
import numpy as np


def get_pixels(folder):
    """
    gets the reflectance data for pixels that will be a part of a constant potential (V)
    Saves the info to a nparray csv file called V_refl.csv
    :param folder: folder of tiff images to grab sections of image to be a part of V from
    :return:  columns and rows of nodes, the intensity values/nodes, number of nodes, number of channels
    """
    images = DWkNNFromROI.make_image_list(folder)

    img1 = images[0]  # cv.cvtColor(image_list[0], cv.COLOR_BGR2GRAY)
    w, h = img1.shape
    rows = []
    cols = []
    channels = len(images)
    intensity = []

    # normalize data, may need to remove
    for i in range(0, len(images)):
        images[i] = images[i] / 255
        # print(images[i])
    locations = []

    roi_list = []
    scale_percent = 500  # percent of original size
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height)
    view_able = cv.resize(img1, dim, interpolation=cv.INTER_AREA)
    while cv.waitKey() != 27:
        region = cv.selectROI("Select Class Pixels", view_able)
        cv.moveWindow("Select Class Pixels", 40, 30)
        scaled = []
        scaled.append(region[0] / (scale_percent / 100))
        scaled.append(region[1] / (scale_percent / 100))
        scaled.append(region[2] / (scale_percent / 100))
        scaled.append(region[3] / (scale_percent / 100))
        roi_list.append(scaled)

    for roi in roi_list:
        for c in range(round(roi[0]), round(roi[0] + roi[2])):
            for r in range(round(roi[1]), round(roi[1] + roi[3])):
                for img in range(0, len(images)):
                    if img == 0:
                        rows.append(r)
                        cols.append(c)
                        locations.append((r, c))
                    intensity.append(images[img][r, c])
    lines = len(locations)
    intensity_np = np.array(intensity)
    intensity_reshaped = intensity_np.reshape(lines, channels)
    np.savetxt("V_refl.csv", intensity_reshaped, delimiter=",")
    return cols, rows, intensity_reshaped, lines, channels, locations


def load_v(filename):
    """
    loads in intensity vales from pixels that are supposed to be part of the potential
    :param filename: the file containing the potential pixel info
    :return: V, the potential
    """
    V = np.loadtxt(filename, delimiter=',')
    return V


def append_and_v(intensities, v_file):
    """
    appends the pixels from a constant potential to pixels in the image
    :param intensities: intensity (reflectance) values from the image
    :param v_file: file to grab reflectance values for potential from
    :return: the combines list of intensities values.
    """
    V = load_v(v_file)
    combo = np.concatenate((intensities, V), axis=0)
    return combo


def make_v(graph, lines):
    """
    Makes a potential based on a const V. :param graph: the kNN graph :param lines: the number of pixels in the
    original image :return: a matrix filled with zeros, except along the diagonal any pixel greater than the number
    of lines (so not in the image) is given a 1 a long the diagonal
    """
    size = graph.shape[0]
    V = np.zeros((size, size))
    for i in range(lines + 1, size):
        V[i, i] = 1
    return V



"""
main for testing
"""
def main():
    get_pixels("imageFiles/selection_cropped_more")
    V = load_v("V_refl.csv")
    print("out")
    print(V)


if __name__ == '__main__':
    main()
