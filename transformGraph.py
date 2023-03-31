"""

Code to perform transform of graph

File: transformGraph.py
Author: Zoe LaLena
Date: 2/7/2023
Course: Senior Project

"""

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from matplotlib import colors
import cv2 as cv

# how many eigen vectors to use
L = 3
T = 46


def graph_laplacian(graph, v):
    """
    get the laplacian of a given graph
    L = D - W
    :param graph: the graph to find the laplacian of
    :return: the laplacian (L) and the diagonal matrix (D)
    """
    alpha = 1
    # get identity
    size = graph.shape[0]
    eye = np.identity(size)

    # get degree of each node
    diagonal_values = np.sum(graph)
    diagonal_mat = eye * diagonal_values
    # print(diagonal_mat)
    laplacian = diagonal_mat - graph
    # v = np.zeros([size, size])
    # v[30,30] = 1
    # v[27,27] = 1
    # v[8,8] = 1

    S = laplacian + alpha * v
    return S, diagonal_mat


def eigen_value_problem(laplacian, diagonal_mat, graph):
    """
    solves the eigen value problem and transforms the graph
    :param laplacian: laplacian of graph
    :param diagonal_mat:  diagonal matrix with degree values of mat
    :param graph: the graph to transform
    :return: the first l eigen vectors and the transformed graph
    """

    eigen_values, eigen_vectors = eigh(laplacian, diagonal_mat, eigvals_only=False)
    eigen_values_sorted_ind = np.argsort(eigen_values, axis=0)
    eigen_values_sorted = np.take_along_axis(eigen_values, eigen_values_sorted_ind, axis=0)
    l_eigen_vectors = np.zeros([graph.shape[0], L])
    for l in range(0, L):
        l_eigen_vectors[:, l] = eigen_vectors[:, eigen_values_sorted_ind[l]]
    transformed_graph = np.dot(l_eigen_vectors.T, graph)
    return l_eigen_vectors, transformed_graph


def detect(transformed_graph, l_eigen_vectors):
    """
    Detects nodes based on threshold, how close to origin nodes are
    :param transformed_graph: transformed graph
    :param l_eigen_vector: first l eigen vectors
    :return:
    """
    magnitudes = []
    size = l_eigen_vectors.shape[0]
    for node in range(0, size):
        mag = 0
        for l in range(1, L):
            mag += pow(l_eigen_vectors[node, l], 2)
        magnitudes.append(math.sqrt(mag))
    mags = np.array(magnitudes)
    detector = 1 / mags
    detected = []
    print("Detected")
    print(detector)
    for i in range(0, size):
        if detector[i] > 1e10:
            detected.append(i)
            print(i)

    print(detector)
    return detected


def display_graph(graph, reflect, l_eigen_vectors):
    """
    Displays first 2 eigen vectors, and labels the pixel # of each pnt (3 channels only)
    :param ind:
    :param graph: kNN graph
    :param reflect: intensities values of each pixel in graph
    :param l_eigen_vectors: first l eigen vectors
    :return:
    """

    fig, ax = plt.subplots()
    for i in range(0, graph.shape[0]):
        #color_val = colors.rgb2hex(reflect[i])
        ax.scatter(l_eigen_vectors[i, 1], l_eigen_vectors[i, 2])#, color=color_val)
        ax.annotate(str(i), (l_eigen_vectors[i, 1], l_eigen_vectors[i, 2]))
    plt.show()


def highlight_pixels(detected, locations, image,lines,save_name):
    """
    We know what pixel numbers have been detected, but not location so lets mark the location on the image
    :return:
    """
    for node in detected:
        if node< lines:
            row = locations[node][0]
            col = locations[node][1]  #
            if row < image.shape[0] and col < image.shape[1]:
                image[row, col] = (255, 255, 0)
    cv.imwrite("imageFiles/" + save_name, image)
    cv.imshow("Result", image)
    cv.waitKey()
    cv.imwrite("result.tif", image)
