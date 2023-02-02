import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import sys
import cv2 as cv


def read_spectra_csv(csv_filepath, band_number):
    """
    Reads the spectral data from a CSV file. The data in the file should be formatted as column locations in the
    first column, rows in the second and then each subsequent column represents the intensity at the that point in one
    of the spectral bands for the set of images being used to form the graph.
    :param csv_filepath: path of csv file to read
    :param band_number: number of bands represented in the given file
    :return: array of column positions, array of row positions, the matrix of the intensity values, number of
    lines in the CSV file/pixels in the image or RIO
    """
    line_num = 0
    coord_col, coord_row, intensity = [], [], []

    with open(csv_filepath, 'r') as csvfile:

        for line in csvfile:
            i = 2
            # split CSV line with a comma
            data = line.split(',')
            # get column and row locations
            coord_col.append(int(data[0]))
            coord_row.append(int(data[1]))
            # after first 2 columns, rest are intensity values.
            while i < band_number + 2:
                intensity.append(float(data[i]))
                i += 1
            # keep track of how many lines are in the file
            line_num += 1
    return np.array(coord_col), np.array(coord_row), np.array(intensity), line_num


def gaussian_weighting_function(theta):
    """
    applies gaussian weighting function to input
    :param theta: input to be weighed by gaussian function
    :return: weighted value
    """
    t = 1  # sigma factor of Gaussian Func
    w = np.exp(-theta / t)
    return w


def create_kNN(data, k):
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


def AdaptiveThreshold(gauss_weighted_graph, k):
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
    #print("adapt thresh: " + str(adapt_thresh))
    adapt_thresh = adapt_thresh / bias

    # adjust the high-end boundary of adapt_thresh
    adapt_thresh[-1] = adapt_thresh[-1] - (adapt_thresh[-1] - adapt_thresh[-2]) / 2

    # adjust the low-end boundary of adapt_thresh
    adapt_thresh[0] = codensity.min() + (adapt_thresh[1] - codensity.min()) / 2

    # get number of Adaptive_Kpix
    #print("adapt thresh: " + str(adapt_thresh))

    # threshold_1, in the low density region
    region_1 = np.where(codensity < adapt_thresh[0])
    region_6 = np.where(codensity > adapt_thresh[-1])
    region_2 = np.where(np.logical_and(codensity >= adapt_thresh[0], codensity < adapt_thresh[1]))
    region_3 = np.where(np.logical_and(codensity >= adapt_thresh[1], codensity < adapt_thresh[2]))
    region_4 = np.where(np.logical_and(codensity >= adapt_thresh[2], codensity < adapt_thresh[3]))
    region_5 = np.where(np.logical_and(codensity >= adapt_thresh[3], codensity <= adapt_thresh[4]))

    return (region_1[0], region_2[0], region_3[0], region_4[0], region_5[0], region_6[0]), adapt_thresh, codensity


def AdaptiveGW(gauss_weighted_graph, pixels_threshold, unweighted_graph):
    """

    :param gauss_weighted_graph: the Gaussian weighted graph adjacency matrix
    :param pixels_threshold: list containing lists of pixels within each threshold
    :param unweighted_graph: unweighted adjacency matrix graph
    :return:
    """

    # define adaptive k values (number of neighbors) based on your scenario
    k_1max = 5  # k_max,here max means distance max, lower density
    k_2 = 6
    k_3 = 8
    k_4 = 10
    k_5 = 12
    k_6min = 15

    # assign adaptive k-values for KNN graph

    # sort unweighted graph
    sorted_unweigh = np.argsort(unweighted_graph)
    #print(sorted_unweigh)
    NN_1max = sorted_unweigh[:, 1:k_1max + 1]
    #print(NN_1max)
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
    return adapt_GW


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


def main():
    if len(sys.argv) != 2:
        print("Command Line Arguments: How many bands?")
        return
    else:
        band_number = int(sys.argv[1])
        coord_col, coord_row, reflect, lineNum = read_spectra_csv('testFiles/3806Test.csv', band_number)
        reflect = reflect.reshape(lineNum,
                                  band_number)
        #print(reflect.shape)
        gauss_weighted_graph, unweighted_graph = create_kNN(reflect, 20)
        pixels_threshold, thresh, coden = AdaptiveThreshold(gauss_weighted_graph, 20)
        adapt_GW = AdaptiveGW(gauss_weighted_graph, pixels_threshold, unweighted_graph)

        # add self-connection
        adapt_GW2 = adapt_GW + np.eye(adapt_GW.shape[0])
        display(adapt_GW2)

    #(thresh)


if __name__ == '__main__':
    main()