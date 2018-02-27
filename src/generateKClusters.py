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
from sklearn import preprocessing

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
        - plotting sse

    • Compare the performance of your algorithm with that of Weka and summarize
      your results. In the report, summarize the differences (if there is any) and
      elaborate on it (why/how). (10 points)

    • Write report

    TODO: - email TA about error output...
          - ask TA if the runtime plots can be separate.. (and what should
            value of k be when testing different dimensions?)
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

    dataframe, classes = parse_arff_file(filename)
    headers = list()
    full_data_averages = list()
    for col in list(dataframe):
        headers.append(col)
        full_data_averages.append(dataframe[col].mean())

    rows, columns = dataframe.shape

    clusters, original_centroids, final_centroids, num_iterations, runtime, error = kmeans.k_means_clustering(dataframe, k, max_iterations, epsilon)
    # plot_goodness_versus_clusters(dataframe)

    print_k_means_data(headers, full_data_averages, rows, clusters, original_centroids, final_centroids, num_iterations, runtime, error, classes)
    # plot_results(clusters)


def plot_runtime_versus_clusters(dataframe):
    k = 1
    max_iterations = 100
    epsilon = 0.01
    max_k = 50

    runtime_arr = []
    while k <= max_k:
        print("Starting iteration for {} clusters.".format(k))
        clusters, original_centroids, final_centroids, num_iterations, runtime, error = kmeans.k_means_clustering(dataframe, k, max_iterations, epsilon)
        runtime_arr.append(runtime)
        k += 1

    x_axis = np.arange(1, max_k+1, 1)
    fig, ax = plt.subplots()
    ax.set_title("Runtime versus Number of Clusters")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Runtime (in seconds)")
    ax.plot(x_axis, runtime_arr, color='r')
    plt.show()


def plot_goodness_versus_clusters(dataframe):
    k = 1
    max_iterations = 100
    epsilon = 0.01
    max_k = 50

    goodness = []
    while k <= max_k:
        print("Starting iteration for {} clusters.".format(k))
        clusters, original_centroids, final_centroids, num_iterations, runtime, error = kmeans.k_means_clustering(dataframe, k, max_iterations, epsilon)
        goodness.append(error)
        k += 1

    x_axis = np.arange(1, max_k+1, 1)
    fig, ax = plt.subplots()
    ax.set_title("Goodness versus Number of Clusters")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("SSE (Goodness)")
    ax.plot(x_axis, goodness, color='r')
    plt.show()


def print_k_means_data(headers, full_data_averages, rows, clusters, original_centroids, final_centroids, num_iterations, runtime, error, classes):
    print("\nNumber of iterations: {}".format(num_iterations))
    print("Within cluster sum of squared errors: {}".format(round(error, 3)))
    print("\nInitial starting points (random):")

    for k,v in original_centroids.items():
        centroid_coordinates = ""
        for item in v:
            centroid_coordinates += (str(round(item, 2)) + ",")
        print("Cluster {}: {}".format(k+1, centroid_coordinates[:-1]))

    print("\nFinal cluster centroids:")
    table_str = "Attribute\tFull Data"
    for i in range(k+1):
        table_str += "\t{}".format(i+1)
    print(table_str)

    table_str = "\t\t({})\t".format(rows)
    for i in range(k+1):
        if i not in clusters:
            table_str += "\t({})".format(0)
        else:
            table_str += "\t({})".format(len(clusters[i]))
    print(table_str)

    print("=========================================================")

    for i in range(len(headers)):
        current_attribute = headers[i].split("@")[0]

        if current_attribute.lower() == 'class':
            row_str = current_attribute + "\t" + str(classes[int(full_data_averages[i])]) + "\t"
            for j in range(k+1):
                row_str += "\t" + str(classes[int(final_centroids[j][i])])
        else:
            row_str = current_attribute + "\t" + str(round(full_data_averages[i], 2)) + "\t"
            for j in range(k+1):
                row_str += "\t" + str(round(final_centroids[j][i], 2))
        print(row_str)

    print("\nTime taken to build model (full training data) : {} seconds".format(round(runtime, 2)))
    
    print("\nClustered Instances")
    for i in range(k+1):
        if i not in clusters:
            print("{}\t{} ({} %)".format(i+1, 0, 0))
        else:
            print("{}\t{} ({} %)".format(i+1, len(clusters[i]), round(((len(clusters[i])/rows)*100), 2)))


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
    classes = []
    with open(filename) as file:
        df = a2p.load(file)

        try:
            df.iloc[:,-1] = df.iloc[:,-1].apply(int)
        except:
            le = preprocessing.LabelEncoder()
            le.fit(df.iloc[:,-1])
            classes = list(le.classes_)
            df.iloc[:,-1] = le.transform(df.iloc[:,-1]) 

        new_df = df.select_dtypes(include=numerics)
        return new_df, classes


if __name__ == '__main__':
    main()