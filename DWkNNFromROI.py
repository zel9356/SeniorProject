"""
Creates a density weighted kNN graph based on a selected region of intrest

File: DWkNNFromROI.py
Author: Zoe LaLena
Date: 2/7/2023
Course: Senior Project
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import sys
import cv2 as cv
import os


def grab_roi(image):
    """
    grabs a region of from an image
    :param image: the image to select a ROI from
    :return:
    """
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
    scaled.append(roi[0] / .5)
    scaled.append(roi[1] / .5)
    scaled.append(roi[2] / .5)
    scaled.append(roi[3] / .5)
    new_image = image[int(scaled[1]):int(scaled[1] + scaled[3]), int(scaled[0]):int(scaled[0] + scaled[2])]

    cv.imshow("Selected Image", new_image)
    cv.waitKey()
    return scaled


def make_image_list(path):
    """
    Makes a list of all the images in a given directory
    :param path:  containing image files
    :return: list of images
    """
    image_names = os.listdir(path)
    # print(image_names)
    images = []
    for image in image_names:
        img = cv.imread(path + "\\" + image, cv.COLOR_BGR2GRAY)
        images.append(img)
    return images


def get_data(images, roi):
    """
    creates array of nodes
    :param images: list of images, 1 image per channel
    :param roi: boundaries of the ROI
    :return: columns and rows of nodes, the intensity values/nodes, number of nodes, number of channels
    """
    rows = []
    cols = []
    lines = int(roi[3]) * int(roi[2])
    channels = len(images)
    intensity = []

    # normalize data, may need to remove
    for i in range(0, len(images)):
        images[i] = images[i] / 255
        # print(images[i])
    locations = []
    for row in range(int(roi[1]), int(roi[1] + roi[3])):
        for col in range(int(roi[0]), int(roi[0] + roi[2])):
            for img in range(0, len(images)):
                if img == 0:
                    rows.append(row)
                    cols.append(cols)
                    locations.append((row,col))
                intensity.append(images[img][row, col])
    intensity_np = np.array(intensity)
    intensity_reshaped = intensity_np.reshape(lines, channels)
    return cols, rows, intensity_reshaped, lines, channels, locations

# def get_data(images, roi):
#     """
#     creates array of nodes
#     :param images: list of images, 1 image per channel
#     :param roi: boundaries of the ROI
#     :return: columns and rows of nodes, the intensity values/nodes, number of nodes, number of channels
#     """
#     rows = []
#     cols = []
#     lines = int(roi[3]) * int(roi[2])
#     channels = len(images)
#     intensity = []
#
#     # normalize data, may need to remove
#     for i in range(0, len(images)):
#         images[i] = images[i] / 255
#         # print(images[i])
#     locations = []
#     pixel_amount = 0
#     for row in range(int(roi[1]), int(roi[1] + roi[3])):
#         for col in range(int(roi[0]), int(roi[0] + roi[2])):
#             pixel_values = []
#             for img in range(0, len(images)):
#                 if img == 0:
#                     rows.append(row)
#                     cols.append(cols)
#                     locations.append((row, col))
#                 pixel_values.append(images[img][row, col])
#             intensity.append(np.array(pixel_values))
#             pixel_amount = pixel_amount + 1
#     intensity_np = np.array(intensity)
#     # intensity_reshaped = intensity_np.reshape(lines, channels)
#     return cols, rows, intensity_np, lines, channels, locations


def gaussian_weighting_function(theta):
    """
    applies gaussian weighting function to input
    :param theta: input to be weighed by gaussian function
    :return: weighted value
    """
    t = 1  # sigma factor of Gaussian Func
    w = np.exp(-theta / t)
    return w


def create_knn(data, k):
    """
    creates a basic kNN graph from
    :param data: intensity data as shape [number of pixels, number of bands]
    :param k: number of neighbors to connect each node/pixel to
    :return: A gaussian weighted and non gaussian weighted graph based on spectral similarity
    """
    number_of_nodes = data.shape[0]
    gauss_weighted_graph = np.zeros((number_of_nodes, number_of_nodes))

    # find the arc cos of the similarity
    # cosine_similarity finds pairwise similarities between all samples in data
    # this will just be number_of_nodes X number_of_nodes matrix, representing the spectral angle between each node
    unweighted_graph = np.arccos(cosine_similarity(data))

    # sort the unweighted graph
    sorted_graph = np.argsort(unweighted_graph)
    # grab the k most similar nodes for each node, excluding self
    sorted_graph_k = sorted_graph[:, 1:k + 1]

    # add k nearest neighbors to the gaussian weighted graph
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if j in sorted_graph_k[i]:
                # apply gaussian weighting function
                gauss_weighted_graph[i][j] = gaussian_weighting_function(unweighted_graph[i, j])

    return gauss_weighted_graph, unweighted_graph


def adaptive_threshold(gauss_weighted_graph, k):
    """
    Based on the z score, finds which pixels fall where within the adpaptive threshold
    :param gauss_weighted_graph: the Gaussian weighted graph
    :param k: the number of neighbors each pixel is connected to
    :return: list containing lists of pixels within each threshold, the final adapted threshold, and the codensity
    """

    # calculate codensity
    codensity = np.sum(gauss_weighted_graph, axis=1) / k

    # assign adaptive threshold based on z-score of codensity
    # decide adaptive k values via z-score of normal distribution
    z_score_list = [-2, -1, 0, +1, +2]
    z_score = np.array(z_score_list)
    mean = np.mean(codensity)
    std = np.std(codensity)

    adapt_thresh = z_score * std + mean

    # introduce a normalization factor by bias
    bias = adapt_thresh[-1] / codensity.max()
    # print("adapt thresh: " + str(adapt_thresh))
    adapt_thresh = adapt_thresh / bias

    # adjust the high-end boundary of adapt_thresh
    adapt_thresh[-1] = adapt_thresh[-1] - (adapt_thresh[-1] - adapt_thresh[-2]) / 2

    # adjust the low-end boundary of adapt_thresh
    adapt_thresh[0] = codensity.min() + (adapt_thresh[1] - codensity.min()) / 2

    # get number of Adaptive_Kpix
    # print("adapt thresh: " + str(adapt_thresh))

    # threshold_1, in the low density region
    region_1 = np.where(codensity < adapt_thresh[0])
    region_6 = np.where(codensity > adapt_thresh[-1])
    region_2 = np.where(np.logical_and(codensity >= adapt_thresh[0], codensity < adapt_thresh[1]))
    region_3 = np.where(np.logical_and(codensity >= adapt_thresh[1], codensity < adapt_thresh[2]))
    region_4 = np.where(np.logical_and(codensity >= adapt_thresh[2], codensity < adapt_thresh[3]))
    region_5 = np.where(np.logical_and(codensity >= adapt_thresh[3], codensity <= adapt_thresh[4]))

    return (region_1[0], region_2[0], region_3[0], region_4[0], region_5[0], region_6[0]), adapt_thresh, codensity


def adaptive_gw(gauss_weighted_graph, pixels_threshold, unweighted_graph):
    """
    Edits gaussian weighted graph to be a density weighted graph
    :param gauss_weighted_graph: the Gaussian weighted graph adjacency matrix
    :param pixels_threshold: list containing lists of pixels within each threshold
    :param unweighted_graph: unweighted adjacency matrix graph
    :return:
    """

    # define adaptive k values (number of neighbors) based on your scenario
    # k_1max = 5  # k_max,here max means distance max, lower density
    # k_2 = 6
    # k_3 = 8
    # k_4 = 10
    # k_5 = 12
    # k_6min = 15
    k_1max = 1  # k_max,here max means distance max, lower density
    k_2 = 2
    k_3 = 3
    k_4 = 4
    k_5 = 5
    k_6min = 6

    # assign adaptive k-values for KNN graph

    # sort unweighted graph
    sorted_unweigh = np.argsort(unweighted_graph)
    # rows = np.array(rows)
    # cols = np.array(cols)
    # cols = np.take_along_axis(cols, sorted_unweigh[:,0], axis=0)
    # rows = np.take_along_axis(rows, sorted_unweigh[:, 0], axis=0)
    # print(sorted_unweigh)
    NN_1max = sorted_unweigh[:, 1:k_1max + 1]
    # print(NN_1max)
    NN_2 = sorted_unweigh[:, 1:k_2 + 1]
    NN_3 = sorted_unweigh[:, 1:k_3 + 1]
    NN_4 = sorted_unweigh[:, 1:k_4 + 1]
    NN_5 = sorted_unweigh[:, 1:k_5 + 1]
    NN_6min = sorted_unweigh[:, 1:k_6min + 1]

    num = len(unweighted_graph)
    adapt_GW = np.zeros((num, num))

    # depend on adaptive k
    for i in range(num):
        if i in pixels_threshold[0]:  # i is in high density region
            for j in range(num):
                if j in NN_1max[i]:
                    adapt_GW[i][j] = gauss_weighted_graph[i, j]
        elif i in pixels_threshold[1]:  # i is in average density region
            for j in range(num):
                if j in NN_2[i]:
                    adapt_GW[i][j] = gauss_weighted_graph[i, j]
        elif i in pixels_threshold[2]:  # i is in average density region
            for j in range(num):
                if j in NN_3[i]:
                    adapt_GW[i][j] = gauss_weighted_graph[i, j]
        elif i in pixels_threshold[3]:  # i is in average density region
            for j in range(num):
                if j in NN_4[i]:
                    adapt_GW[i][j] = gauss_weighted_graph[i, j]
        elif i in pixels_threshold[4]:  # i is in average density region
            for j in range(num):
                if j in NN_5[i]:
                    adapt_GW[i][j] = gauss_weighted_graph[i, j]
        else:  # i is in low density region
            for j in range(num):
                if j in NN_6min[i]:
                    adapt_GW[i][j] = gauss_weighted_graph[i, j]
    return adapt_GW, sorted_unweigh


# Plot adjacency matrix
def display(adj_mat):
    """
    Displays the adjacency matrix as a graph
    :param adj_mat: matrix to display
    :return:
    """
    plt.imshow(adj_mat, cmap='hot')
    plt.colorbar()
    plt.show()
