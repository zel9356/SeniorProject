"""
main file for creating and displaying a density weighted kNN graph

File: dwkNNMain.py
Author: Zoe LaLena
Date: 2/8/2023
Course: Senior Project
"""

import sys
import DWkNNFromROI
import numpy as np


def main():
    if len(sys.argv) != 2:
        print("Command Line Arguments: path to folder of images")
        return
    else:
        image_list = DWkNNFromROI.make_image_list(sys.argv[1])
        # roi = grabROI(image_list[9])
        # roi = [3806.0, 3546.0, 84.0, 60.0]
        img1 = image_list[0]
        roi = [0, 0, img1.shape[1], img1.shape[0]]
        coord_col, coord_row, reflect, line_num, bands, locations = DWkNNFromROI.get_data(image_list, roi)
        gauss_weighted_graph, unweighted_graph = DWkNNFromROI.create_knn(reflect, 20)
        pixels_threshold, thresh, coden = DWkNNFromROI.adaptive_threshold(gauss_weighted_graph, 20)
        adapt_GW = DWkNNFromROI.adaptive_gw(gauss_weighted_graph, pixels_threshold, unweighted_graph)

        # add self-connection
        adapt_GW2 = adapt_GW + np.eye(adapt_GW.shape[0])
        DWkNNFromROI.display(adapt_GW2)


if __name__ == '__main__':
    main()
