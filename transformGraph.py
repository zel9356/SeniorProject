"""

Author: Zoe LaLena
Date: 2/7/2023
Course: Senior Project

"""
import math

import matplotlib.pyplot as plt
import numpy as np
import DWkNNFromROI
import sys
from scipy.linalg import eigh

def graph_laplacian(graph):
    """
    get the laplacian of a given graph
    L = D - W
    :param graph: the graph to find the laplacian of
    :return: the laplacian (L) and the diagonal matrix (D)
    """
    alpha = 10
    # get identity
    size = graph.shape[0]
    eye = np.identity(size)

    # get degree of each node
    diagonal_values = np.sum(graph)
    diagonal_mat = eye * diagonal_values
    #print(diagonal_mat)
    laplacian = diagonal_mat - graph
    v = np.zeros([size,size])
    v[31,31] = 1
    v[17,17] = 1
    #v[29,29] = 1

    S = laplacian + alpha*v
    return S, diagonal_mat


def eigen_value_problem(laplacian, diagonal_mat,graph,reflect):
    L = 2
    from matplotlib import colors
    eigen_values, eigen_vectors = eigh(laplacian,diagonal_mat, eigvals_only=False)
    eigen_values_sorted_ind = np.argsort(eigen_values, axis=0)
    eigen_values_sorted = np.take_along_axis(eigen_values, eigen_values_sorted_ind, axis=0)
    l_eigen_vectors = np.zeros([graph.shape[0],L])
    for l in range(0,L):
        l_eigen_vectors[:, l] = eigen_vectors[:, eigen_values_sorted_ind[l]]
    transformed_graph = np.dot(l_eigen_vectors.T, graph)
    #print(eigen_values_sorted)
    magnitudes = []
    for node in range(0, graph.shape[0]):
        magnitudes.append(math.sqrt(pow(l_eigen_vectors[node,0],2) + pow(l_eigen_vectors[node,1],2)))
    mags = np.array(magnitudes)
    detector = 1/mags
    for i in range(0,graph.shape[0]):
        if detector[i] >1e16:
            print(i)
    print(detector)
    fig,ax = plt.subplots()
    for i in range(0,graph.shape[0]):
        color_val = colors.rgb2hex(reflect[i])
        ax.scatter(l_eigen_vectors[i,0], l_eigen_vectors[i,1], color = color_val)
        ax.annotate(str(i),(l_eigen_vectors[i,0], l_eigen_vectors[i,1]))
    plt.show()
def main():
    if len(sys.argv) != 2:
        print("Command Line Arguments: path to folder of images")
        return
    else:
        image_list = DWkNNFromROI.makeImageList(sys.argv[1])
        #roi = grabROI(image_list[9])

        img1 = image_list[0]
        roi = [0,0, img1.shape[1],img1.shape[0]]

        # put image data in to matrix node format
        coord_col, coord_row, reflect, lineNum, bands = DWkNNFromROI.getData(image_list, roi )

        # get normal kNN graphs both weighted by gaussian function and not
        gauss_weighted_graph, unweighted_graph = DWkNNFromROI.create_kNN(reflect, 20)

        # find thresholds for "k regions"
        pixels_threshold, thresh, coden = DWkNNFromROI.AdaptiveThreshold(gauss_weighted_graph, 20)

        # based on thresholds, use correct k values
        graph = DWkNNFromROI.AdaptiveGW(gauss_weighted_graph, pixels_threshold, unweighted_graph)

        # add self-connection
        #graph = adapt_GW + np.eye(adapt_GW.shape[0])

        laplacian, diagonal_mat = graph_laplacian(graph)
        eigen_value_problem(laplacian, diagonal_mat, graph,reflect)

if __name__ == '__main__':
    main()
