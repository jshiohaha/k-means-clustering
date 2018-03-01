import sys
import copy
import math
import time
import random
import pandas as pd
import numpy as np


def k_means_clustering(dataframe, k, max_iterations, epsilon, seed):
    num_iterations = 1
    rows, columns = dataframe.shape
    old_sse, new_sse = sys.maxsize, 0
    min_max = [(dataframe[column].min(), dataframe[column].max()) for column in list(dataframe)]

    np.random.seed(seed)
    centroids = {
        i: [np.random.uniform(min_max[idx][0], min_max[idx][1]) for idx in range(len(list(dataframe)))]
        for i in range(k)
    }

    print(">> Created {} random centroids.".format(k))

    original_centroids = copy.deepcopy(centroids)

    start = time.time()
    print("starting time: {}".format(start))
    while(num_iterations <= max_iterations):
        clusters = {}

        for instance in dataframe.values:
            min_dist, cluster_id = dist(instance, centroids)
            if cluster_id not in clusters.keys():
                clusters[cluster_id] = list()
            clusters[cluster_id].append(instance)

        old_sse = new_sse
        new_sse = 0
        for k,list_of_instances in clusters.items():
            centroid_sum = np.zeros(columns)
            centroid_size = len(list_of_instances)

            for instance in list_of_instances:
                centroid_sum = np.add(centroid_sum, instance)

            centroids[k] = [(centroid_sum[i] / centroid_size) for i in range(len(centroid_sum))]
            new_sse += np.sum([np.linalg.norm(instance-centroids[k]) for instance in list_of_instances])

        print("Sum squared errors on {}-th iteration: {}".format(num_iterations, new_sse))

        if(math.fabs(old_sse - new_sse) < epsilon):
            end = time.time()
            print(">> K-means clustering converged because difference in SSE between iteration {} and iteration {} was {}".format(num_iterations,num_iterations+1,math.fabs(old_sse - new_sse)))
            print(">> Ending k-means at {}. Elapsed time was {}.".format(end, end-start))
            return clusters, original_centroids, centroids, num_iterations, (end-start), new_sse

        num_iterations += 1

    print(">> Reached max number of {} iterations before stopping iteration on k means clustering...".format(max_iterations))
    end = time.time()
    print(">> Ending k-means at {}. Elapsed time was {}.".format(end, end-start))
    return clusters, original_centroids, centroids, num_iterations, (end-start), new_sse


def dist(instance, centroids):
    min_dist = sys.maxsize
    min_dist_key = 0

    for k,v in centroids.items():
        d = np.linalg.norm(np.array(instance).reshape(1,-1)-np.array(v).reshape(1,-1))

        if d < min_dist:
            min_dist = d
            min_dist_key = k

    return min_dist, min_dist_key
