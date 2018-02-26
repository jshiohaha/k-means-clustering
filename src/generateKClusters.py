import sys
import csv
import json
import time
import pprint

import fileUtils as fileUtils
import kmeans as kmeans

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


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

    dataframe = fileUtils.parse_arff_file(filename)
    rows, columns = dataframe.shape

    clusters, centroids = kmeans.k_means_clustering(dataframe, k, max_iterations, epsilon)

    for k,v in centroids.items():
        print("Cluster {} centroid: {}".format(k, v))

    # print the percentage of instances in each cluster
    for k,v in clusters.items():
        print("cluster {}: {}%".format(k, ((len(v)/rows)*100)))

    plot_results(clusters)


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


if __name__ == '__main__':
    main()