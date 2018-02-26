import sys
import copy
import math
import timeit

import pprint as pprint
import numpy as np

from scipy.spatial import distance


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

    print(">> Created {} random centroids.".format(k))

    # create a numpy array from the dataframe
    numpy_arr_of_instances = dataframe.values

    start = timeit.timeit()
    print("starting time: {}".format(start))
    while(num_iterations < max_iterations):
        clusters = {}

        # calculate distance to nearest cluster, and assign
        for instance in numpy_arr_of_instances:
            # remove the index of each instance
            distances = dist(instance, centroids)

            min_dist = min(distances)
            cluster_num = distances.index(min_dist)
            # print_distance_info(instance, distances)
            # print("Assigning instance to centroid {}".format(cluster_num))

            if cluster_num not in clusters.keys():
                clusters[cluster_num] = list()
            clusters[cluster_num].append(instance)

        # find new cluster centers by taking the average of all assigned points
        for k,list_of_instances in clusters.items():
            centroid_sum = np.zeros(columns)
            centroid_size = len(list_of_instances)
            for instance in list_of_instances:
                for i in range(len(instance)):
                    centroid_sum[i] += instance[i]
            # assigning average of the i-th attribute to the i-th attribute of the k-th centroid
            centroids[k] = [(centroid_sum[i] / centroid_size) for i in range(len(centroid_sum))]

        old_sse = new_sse
        new_sse = 0
        # Calculate the sum squared error on the current iteration
        for k,list_of_instances in clusters.items():
            for instance in list_of_instances:
                new_sse += distance.euclidean(centroids[k], instance)

        print("Sum squared errors on {}-th iteration: {}".format(num_iterations+1, new_sse))

        if(math.fabs(old_sse - new_sse) < epsilon):
            end = timeit.timeit()
            print("ending time: {}".format(end))
            print("elapsed time: {}".format(end-start))

            print(">> K-means clustering converged because difference in SSE between iteration {} and iteration {} was {}".format(num_iterations+1,num_iterations+2,math.fabs(old_sse - new_sse)))
            return clusters, original_centroids, centroids, num_iterations, (-start)

        num_iterations += 1

    end = timeit.timeit()
    print("ending time: {}".format(end))
    print("elapsed time: {}".format(end-start))

    print(">> Reached max number of {} iterations before stopping iteration on k means clustering...".format(max_iterations))
    return clusters, original_centroids, centroids, num_iterations, (timeit.timeit()-start)


def dist(a, centroids):
    ''' a is the instance of data, and centroids is dictionary of k-centroids

        todo: validate this returns the correct minimum distance
    '''
    distances = []
    for k,v in centroids.items():
        distances.append(distance.euclidean(a, v))
    return distances


def print_distance_info(instance, distances):
    curr_str = "{}\t{}".format(instance[0], instance[1])
        
    for d in distances:
        curr_str += "\t{}".format(round(d,2))

    min_dist = min(distances)
    min_dist_idx = distances.index(min_dist)
    curr_str += "\t{}".format(min_dist_idx)

    print(curr_str)
