"""

Main function to create graph, transform it and detect

File: transformAndDetect.py
Author: Zoe LaLena
Date: 2/8/2023
Course: Senior Project

"""

import sys
import DWkNNFromROI
import transformGraph
import cv2 as cv


def main():
    if len(sys.argv) != 2:
        print("Command Line Arguments: path to folder of images")
        return
    else:
        image_list = DWkNNFromROI.make_image_list(sys.argv[1])
        # roi = grabROI(image_list[9])

        img1 = image_list[0]
        roi = [0, 0, img1.shape[1], img1.shape[0]]

        # put image data in to matrix node format
        coord_col, coord_row, reflect, number_of_lines, bands, locations = DWkNNFromROI.get_data(image_list, roi)

        # get normal kNN graphs both weighted by gaussian function and not
        gauss_weighted_graph, unweighted_graph = DWkNNFromROI.create_knn(reflect, 20)

        # find thresholds for "k regions"
        pixels_threshold, thresh, coden = DWkNNFromROI.adaptive_threshold(gauss_weighted_graph, 20)

        # based on thresholds, use correct k values
        graph = DWkNNFromROI.adaptive_gw(gauss_weighted_graph, pixels_threshold, unweighted_graph)

        # add self-connection
        # graph = adapt_GW + np.eye(adapt_GW.shape[0])

        laplacian, diagonal_mat = transformGraph.graph_laplacian(graph)
        l_eigen_vectors, transformed_graph = transformGraph.eigen_value_problem(laplacian, diagonal_mat, graph)
        detected_numbers = transformGraph.detect(transformed_graph, l_eigen_vectors)
        transformGraph.display_graph(graph, reflect, l_eigen_vectors)
        # grab orginal image for testing
        image_color = cv.imread("imageFiles/smallImage.png")
        transformGraph.highlight_pixels(detected_numbers,locations,image_color)


if __name__ == '__main__':
    main()
