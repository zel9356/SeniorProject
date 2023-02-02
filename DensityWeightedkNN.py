import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import sys


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
    # calculate codensity
    sum_dist = np.sum(gauss_weighted_graph, axis=1)
    codensity = sum_dist / k

    # assign adaptive threshold based on z-score of codensity
    # decide adaptive k values via z-score of normal distribution
    z_score = [-2, -1, 0, +1, +2]

    mean = np.mean(codensity)
    std = np.std(codensity)

    adapt_thresh = [i * std + mean for i in z_score]

    # introduce a normalization factor by bias
    bias = adapt_thresh[-1] / codensity.max()

    adapt_thresh = adapt_thresh / bias

    # adjust the high-end boundary of adapt_thresh
    adapt_thresh[-1] = adapt_thresh[-1] - (adapt_thresh[-1] - adapt_thresh[-2]) / 2

    # adjust the low-end boundary of adapt_thresh
    adapt_thresh[0] = codensity.min() + (adapt_thresh[1] - codensity.min()) / 2

    # get number of Adaptive_Kpix

    # threshold_1, in the low density region
    k_1 = np.where(codensity < adapt_thresh[0])
    print(len(k_1[0]))

    # threshold_2: interval
    k_2left = np.where(codensity > adapt_thresh[0])

    dict_k2left = {}
    for i in range(len(k_2left[0])):
        dict_k2left[k_2left[0][i]] = codensity[k_2left[0][i]]

    k_2right = np.where(codensity < adapt_thresh[1])

    dict_k2right = {}
    for i in range(len(k_2right[0])):
        dict_k2right[k_2right[0][i]] = codensity[k_2right[0][i]]

    dict_k2 = dict_k2left.keys() & dict_k2right.keys()
    dict_k2 = np.sort(list(dict_k2))  # convert set to list and then sort

    print(len(dict_k2))

    # threshold_3: interval
    k_3left = np.where(codensity > adapt_thresh[1])

    dict_k3left = {}
    for i in range(len(k_3left[0])):
        dict_k3left[k_3left[0][i]] = codensity[k_3left[0][i]]

    k_3right = np.where(codensity < adapt_thresh[2])

    dict_k3right = {}
    for i in range(len(k_3right[0])):
        dict_k3right[k_3right[0][i]] = codensity[k_3right[0][i]]

    dict_k3 = dict_k3left.keys() & dict_k3right.keys()
    dict_k3 = np.sort(list(dict_k3))  # convert set to list and then sort

    print(len(dict_k3))

    # threshold_4: interval
    k_4left = np.where(codensity > adapt_thresh[2])

    dict_k4left = {}
    for i in range(len(k_4left[0])):
        dict_k4left[k_4left[0][i]] = codensity[k_4left[0][i]]

    k_4right = np.where(codensity < adapt_thresh[3])

    dict_k4right = {}
    for i in range(len(k_4right[0])):
        dict_k4right[k_4right[0][i]] = codensity[k_4right[0][i]]

    dict_k4 = dict_k4left.keys() & dict_k4right.keys()
    dict_k4 = np.sort(list(dict_k4))  # convert set to list and then sort

    print(len(dict_k4))

    # threshold_5: interval
    k_5left = np.where(codensity > adapt_thresh[3])

    dict_k5left = {}
    for i in range(len(k_5left[0])):
        dict_k5left[k_5left[0][i]] = codensity[k_5left[0][i]]

    k_5right = np.where(codensity < adapt_thresh[4])

    dict_k5right = {}
    for i in range(len(k_5right[0])):
        dict_k5right[k_5right[0][i]] = codensity[k_5right[0][i]]

    dict_k5 = dict_k5left.keys() & dict_k5right.keys()
    dict_k5 = np.sort(list(dict_k5))  # convert set to list and then sort

    print(len(dict_k5))

    # threshold_6: interval
    k_6 = np.where(codensity > adapt_thresh[4])

    print(len(k_6[0]))

    return (k_1[0], dict_k2, dict_k3, dict_k4, dict_k5, k_6[0]), adapt_thresh, codensity


def AdaptiveGW(GW, Kpixel, OriW):
    # define adaptive k values (number of neighbors) based on your scenario
    k_1max = 5  # k_max,here max means distance max, lower density
    k_2 = 6
    k_3 = 8
    k_4 = 10
    k_5 = 12
    k_6min = 15

    # assign adaptive k-values for KNN graph
    NN = np.argsort(OriW)
    NN_1max = NN[:, 1:k_1max + 1]
    NN_2 = NN[:, 1:k_2 + 1]
    NN_3 = NN[:, 1:k_3 + 1]
    NN_4 = NN[:, 1:k_4 + 1]
    NN_5 = NN[:, 1:k_5 + 1]
    NN_6min = NN[:, 1:k_6min + 1]

    num = len(OriW)
    adapt_GW = np.zeros((num, num))

    # depend on adaptive k
    for i in range(num):
        if i in Kpixel[0]:  # i is in high density region
            for j in range(num):
                if j in NN_1max[i]:
                    adapt_GW[i][j] = GW[i, j]
        elif i in Kpixel[1]:  # i is in average density region
            for j in range(num):
                if j in NN_2[i]:
                    adapt_GW[i][j] = GW[i, j]
        elif i in Kpixel[2]:  # i is in average density region
            for j in range(num):
                if j in NN_3[i]:
                    adapt_GW[i][j] = GW[i, j]
        elif i in Kpixel[3]:  # i is in average density region
            for j in range(num):
                if j in NN_4[i]:
                    adapt_GW[i][j] = GW[i, j]
        elif i in Kpixel[4]:  # i is in average density region
            for j in range(num):
                if j in NN_5[i]:
                    adapt_GW[i][j] = GW[i, j]
        else:  # i is in low density region
            for j in range(num):
                if j in NN_6min[i]:
                    adapt_GW[i][j] = GW[i, j]
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
        coord_col, coord_row, reflect, lineNum = read_spectra_csv('filename.csv', band_number)
        reflect = reflect.reshape(lineNum,
                                  band_number)
        print(reflect.shape)
        GW, W = create_kNN(reflect, 20)
        Kpixel, thresh, coden = AdaptiveThreshold(GW, 20)
        adapt_GW = AdaptiveGW(GW, Kpixel, W)
        # adapt_GW.shape
        # add self-connection
        adapt_GW2 = adapt_GW + np.eye(adapt_GW.shape[0])
        display(adapt_GW2)

    print(thresh)


if __name__ == '__main__':
    main()