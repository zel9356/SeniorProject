"""

text classification with set v

File: test_with_set_v.py
Author: Zoe LaLena
Date: 3/8/2023
Course: Senior Project

"""

import sys
import DWkNNFromROI
import transformGraph
import cv2 as cv
import makeVMatrix
import presetV
import os


def testWithSetV(set_v, folder_of_imgs, save_name):

    image_list = DWkNNFromROI.make_image_list(folder_of_imgs)
    image_names = os.listdir(folder_of_imgs)
    # roi = grabROI(image_list[9])
    # print(image_names)
    img1 = image_list[0]  # cv.cvtColor(image_list[0], cv.COLOR_BGR2GRAY)
    roi = [0, 0, img1.shape[1], img1.shape[0]]

    # put image data in to matrix node format
    coord_col, coord_row, reflect, number_of_lines, bands, locations = DWkNNFromROI.get_data(image_list, roi)
    reflect = presetV.append_and_v(reflect, set_v)
    # get normal kNN graphs both weighted by gaussian function and not
    gauss_weighted_graph, unweighted_graph = DWkNNFromROI.create_knn(reflect, 20)

    # find thresholds for "k regions"
    pixels_threshold, thresh, coden = DWkNNFromROI.adaptive_threshold(gauss_weighted_graph, 20)

    # based on thresholds, use correct k values
    graph, ind = DWkNNFromROI.adaptive_gw(gauss_weighted_graph, pixels_threshold, unweighted_graph)
    # add self-connection
    # graph = adapt_GW + np.eye(adapt_GW.shape[0])
    V = presetV.make_v(graph, number_of_lines)
    laplacian, diagonal_mat = transformGraph.graph_laplacian(graph, V)
    l_eigen_vectors, transformed_graph = transformGraph.eigen_value_problem(laplacian, diagonal_mat, graph)
    detected_numbers,detector = transformGraph.detect(transformed_graph, l_eigen_vectors)
    transformGraph.produce_detection_image(number_of_lines, locations,detector,img1,save_name)
    transformGraph.display_graph(graph, reflect, l_eigen_vectors)
    # grab original image for testing
    image_color = cv.imread(folder_of_imgs + "/" + image_names[0])
    transformGraph.highlight_pixels(detected_numbers, locations, image_color, number_of_lines,
                                    save_name + image_names[0])
