import sys
import csv
import json
import time
import pprint

import kmeans as kmeans

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arff2pandas import a2p
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

'''
    TODO: 
    • Plot the runtime of the algorithm as a function of number of clusters,
      number of dimensions and size of the dataset (number of transactions).
      In the report, you should write a paragraph to summarize the observation
      and elaborate on it. (10 points)

    • Plot the goodness of clustering as a function of the number of clusters
      and determine the optimal number of clusters. In the report, you should
      write a paragraph to summarize the observation and elaborate on it. 
      (15 points)

    • Compare the performance of your algorithm with that of Weka and summarize
      your results. In the report, summarize the differences (if there is any) and
      elaborate on it (why/how). (10 points)

    • Write report



    Current output -- mostly matches weka: 

        Number of iterations: 3
        Within cluster sum of squared errors: 1

        Initial starting points (random):
        Cluster 1: 7.32,2.99,3.78,1.35
        Cluster 2: 5.78,2.01,4.41,2.08
        Cluster 3: 5.81,3.44,1.62,0.99

        Final cluster centroids:
        Attribute   Full Data   1   2   3
                (150)       (39)    (61)    (50)
        =========================================================
        sepallength 5.84        6.85    5.88    5.01
        sepalwidth  3.05        3.08    2.74    3.42
        petallength 3.76        5.72    4.39    1.46
        petalwidth  1.2     2.05    1.43    0.24

        Time taken to build model (full training data) : 0.0 seconds

        Clustered Instances
        1   39 (26.0 %)
        2   61 (40.67 %)
        3   50 (33.33 %)
'''


def main():
    filename = ''
    k = 0
    epsilon = 0
    max_iterations = 0

    arg_length = len(sys.argv)

    if arg_length != 9:
        print("Incorrect number of CLI arguments. Please see README for an example command for how to run this program.")
        sys.exit()
    else:
        if '-f' in sys.argv:
            idx = sys.argv.index('-f')

            if isinstance(sys.argv[idx+1], str):
                input_filename = sys.argv[idx+1]

                user_file = Path(input_filename)
                if not user_file.exists() or not user_file.is_file():
                    print("Filename: {} does not exist. Exiting...".format(user_file))
                    sys.exit()

                filename = input_filename
                print("Using the specified value for input filename: " + str(input_filename))
            else:
                print("Incorrect paramter specification. Exiting...")
                sys.exit()

        if '-k' in sys.argv:
            idx = sys.argv.index('-k')

            if isinstance(sys.argv[idx+1], str):
                try:
                    k = int(sys.argv[idx+1])
                    print("Using the specified value for k: " + str(k) + ".")
                except:
                    print("Could not parse {} into an integer.".format(sys.argv[idx+1]))
            else:
                print("Incorrect paramter specification. Exiting...")
                sys.exit()

        if '-e' in sys.argv:
            idx = sys.argv.index('-e')

            if isinstance(sys.argv[idx+1], str):
                try:
                    epsilon = float(sys.argv[idx+1])
                    print("Using the specified value for epsilon: " + str(epsilon) + ".")
                except:
                    print("Could not parse {} into a float.".format(sys.argv[idx+1]))
            else:
                print("Incorrect paramter specification. Exiting...")
                sys.exit()

        if '-i' in sys.argv:
            idx = sys.argv.index('-i')

            if isinstance(sys.argv[idx+1], str):
                try:
                    max_iterations = int(sys.argv[idx+1])
                    print("Using the specified value for max iterations: " + str(max_iterations) + ".")
                except:
                    print("Could not parse {} into an integer.".format(sys.argv[idx+1]))
            else:
                print("Incorrect paramter specification. Exiting...")
                sys.exit()

    dataframe = parse_arff_file(filename)

    headers = list()
    full_data_averages = list()
    for col in list(dataframe):
        headers.append(col)
        full_data_averages.append(dataframe[col].mean())

    rows, columns = dataframe.shape

    clusters, original_centroids, final_centroids, num_iterations, runtime = kmeans.k_means_clustering(dataframe, k, max_iterations, epsilon)

    print_k_means_data(headers, full_data_averages, rows, clusters, original_centroids, final_centroids, num_iterations, runtime)

    # plot_results(clusters)


def print_k_means_data(headers, full_data_averages, rows, clusters, original_centroids, final_centroids, num_iterations, runtime):
    print("\nNumber of iterations: {}".format(num_iterations))
    # TODO: what is the equivalent of this from Weka?
    print("Within cluster sum of squared errors: {}".format(1))
    print("\nInitial starting points (random):")

    for k,v in original_centroids.items():
        centroid_coordinates = ""
        for item in v:
            centroid_coordinates += (str(round(item, 2)) + ",")
        print("Cluster {}: {}".format(k+1, centroid_coordinates[:-1]))

    print("\nFinal cluster centroids:")
    table_str = "Attribute\tFull Data"
    for i in range(len(final_centroids)):
        table_str += "\t{}".format(i+1)
    print(table_str)

    table_str = "\t\t({})\t".format(rows)
    for i in range(len(clusters)):
        table_str += "\t({})".format(len(clusters[i]))
    print(table_str)

    print("=========================================================")

    for i in range(len(headers)):
        row_str = headers[i].split("@")[0] + "\t" + str(round(full_data_averages[i], 2)) + "\t"
        for j in range(k+1):
            row_str += "\t" + str(round(final_centroids[j][i], 2))
        print(row_str)

    print("\nTime taken to build model (full training data) : {} seconds".format(round(runtime, 2)))
    
    print("\nClustered Instances")
    for k,v in clusters.items():
        print("{}\t{} ({} %)".format(k+1, len(v), round(((len(v)/rows)*100), 2)))


def plot_results(clusters, features=['sepallength', 'sepalwidth', 'petallength']):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    species_arr = ['setosa', 'versicolor', 'virginica']
    colors = ['green', 'red', 'blue']

    for k,v in clusters.items():
        x_plt, y_plt, z_plt = [],[],[]

        for instance in v:
            x_plt.append(instance[0])
            y_plt.append(instance[1])
            z_plt.append(instance[2])
        print("Added {} instances to cluster {} to print in color {}.".format(len(x_plt),k,colors[k]))
        ax.scatter(x_plt, y_plt, z_plt, color=colors[k])

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])

    ax.legend()
    plt.show()


def parse_arff_file(filename):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    with open(filename) as file:
        df = a2p.load(file)
        new_df = df.select_dtypes(include=numerics)
        return new_df


if __name__ == '__main__':
    main()