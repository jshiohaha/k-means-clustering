import sys
import copy
import math
import time
import collections

import pprint as pprint
import numpy as np

from scipy.spatial import distance
from scipy.spatial.distance import cdist


def k_means_clustering(dataframe, k, max_iterations, epsilon):
    num_iterations = 0
    rows, columns = dataframe.shape
    old_sse, new_sse = sys.maxsize, 0

    # generate k initial centroids with each coordinate between the min and max values for a specific column
    # the dataframe
    dataframe_as_list = list(dataframe)
    min_max = [(dataframe[column].min(), dataframe[column].max()) for column in dataframe_as_list]

    centroids = {
        i: [np.random.uniform(min_max[idx][0], min_max[idx][1]) for idx in range(len(dataframe_as_list))]
        for i in range(k)
    }

    original_centroids = copy.deepcopy(centroids)

    # print(">> Created {} random centroids.".format(k))

    # create a numpy array from the dataframe
    numpy_arr_of_instances = dataframe.values

    start = time.time()
    # print("starting time: {}".format(start))
    while(num_iterations < max_iterations):
        clusters = {}

        # calculate distance to nearest cluster, and assign to tht cluster
        # Grows linearly with number of instances because we iterate through all
        # transactions to assign to a cluster
        for instance in numpy_arr_of_instances:
            distances, min_dist, cluster_id = dist(instance, centroids)

            if cluster_id not in clusters.keys():
                clusters[cluster_id] = list()
            clusters[cluster_id].append(instance)

        # find new cluster centers by taking the average of all assigned points
        # todo... improve performance here
        for k,list_of_instances in clusters.items():
            centroid_sum = np.zeros(columns)
            centroid_size = len(list_of_instances)

            for instance in list_of_instances:
                centroid_sum = np.add(centroid_sum, instance)
            # assigning average of the i-th attribute to the i-th attribute of the k-th centroid

            centroids[k] = [(centroid_sum[i] / centroid_size) for i in range(len(centroid_sum))]

        # print("new centroids...")
        # pprint.pprint(centroids)

        old_sse = new_sse
        new_sse = 0
        # Calculate the sum squared error on the current iteration
        for k,list_of_instances in clusters.items():
            # print("For cluster {}, we have {} instances.".format(k, len(list_of_instances)))
            for instance in list_of_instances:
                new_sse += np.linalg.norm(np.subtract(centroids[k], instance))

        # print("Sum squared errors on {}-th iteration: {}".format(num_iterations+1, new_sse))

        if(math.fabs(old_sse - new_sse) < epsilon):
            # print(">> K-means clustering converged because difference in SSE between iteration {} and iteration {} was {}".format(num_iterations+1,num_iterations+2,math.fabs(old_sse - new_sse)))
            end = time.time()
            # print(">> Ending k-means at {}. Elapsed time was {}.".format(end, end-start))
            return clusters, original_centroids, centroids, num_iterations, (end-start), new_sse

        num_iterations += 1

    # print(">> Reached max number of {} iterations before stopping iteration on k means clustering...".format(max_iterations))
    end = time.time()
    # print(">> Ending k-means at {}. Elapsed time was {}.".format(end, end-start))
    return clusters, original_centroids, centroids, num_iterations, (end-start), new_sse


def dist(a, centroids):
    ''' a is the instance of data, and centroids is dictionary of k-centroids

        todo: validate this returns the correct minimum distance
    '''
    distances = []
    min_dist = sys.maxsize
    min_dist_key = 0
    a = np.array(a).reshape(1,-1)

    for k,v in centroids.items():
        v = np.array(v).reshape(1,-1)
        d = cdist(a, v, 'euclidean')

        if d < min_dist:
            min_dist = d
            min_dist_key = k
        distances.append(d)

    return distances, min_dist, min_dist_key


def print_distance_info(instance, distances):
    curr_str = "{}\t{}".format(instance[0], instance[1])
        
    for d in distances:
        curr_str += "\t{}".format(round(d,2))

    min_dist = min(distances)
    min_dist_idx = distances.index(min_dist)
    curr_str += "\t{}".format(min_dist_idx)

    print(curr_str)
