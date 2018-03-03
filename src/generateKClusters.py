import sys
import csv
import json
import time
import pprint
import copy

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
    • Plot the runtime of the algorithm as a function of number of dimensions.
    In the report, you should write a paragraph to summarize the observation
    and elaborate on it. (10 points)

    can we normalize the dataset but preserve the original points by keeping the indices
    of the values?
'''


def main():
    filename, k, epsilon, max_iterations, seed, normalize = parse_command_line_args(sys.argv)
    dataframe, classes, original_dataframe = parse_arff_file(filename, normalize)

    rows, columns = dataframe.shape
    clusters, original_centroids, final_centroids, num_iterations, runtime, error = kmeans.k_means_clustering(dataframe, k, max_iterations, epsilon, seed)

    print_k_means_data(original_dataframe, clusters, original_centroids, final_centroids, num_iterations, runtime, error, classes, normalize)
    # plot_results(clusters)
    # plot_runtime_versus_clusters(dataframe, seed)
    # plot_runtime_versus_dimensions()


def plot_runtime_versus_clusters(dataframe, seed):
    k = 1
    max_iterations = 100
    epsilon = 0.01
    max_k = 25

    runtime_arr = []
    while k <= max_k:
        print("Starting iteration for {} clusters.".format(k))
        clusters, original_centroids, final_centroids, num_iterations, runtime, error = kmeans.k_means_clustering(dataframe, k, max_iterations, epsilon, seed)
        runtime_arr.append(runtime)
        k += 1

    x_axis = np.arange(1, max_k+1, 1)
    fig, ax = plt.subplots()
    ax.set_title("Runtime versus Number of Clusters")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Runtime (in seconds)")
    ax.plot(x_axis, runtime_arr, color='r')
    plt.show()


def plot_runtime_versus_dimensions():
    ''' Used Weka EM Clustering Algorithm to 
        use cross validation to determine the
        optimal k for values_of_k corresponding
        to the value of k to use for each of the
        datsets.
    '''
    files = ['../Data/dimensions/train-5-1000.arff',
             '../Data/dimensions/train-10-1000.arff',
             '../Data/dimensions/train-12-1000.arff',
             '../Data/dimensions/train-40-1000.arff',
             '../Data/dimensions/train-784-1000.arff',
             '../Data/dimensions/train-10000-100.arff']
    max_iterations = 5
    epsilon = 0.01
    max_k = 25
    values_of_k = [12,5,19,7,5,15]
    runtime_arr = []
    num_dimensions = []

    for idx,file in enumerate(files):
        dataframe, classes, original_dataframe = parse_arff_file(file, normalize=True)
        headers = list()
        full_data_averages = list()
        for col in list(dataframe):
            headers.append(col)
            full_data_averages.append(dataframe[col].mean())

        rows, columns = dataframe.shape
        clusters, original_centroids, final_centroids, num_iterations, runtime, error = kmeans.k_means_clustering(dataframe, values_of_k[idx], max_iterations, 0.01, seed=None)
        runtime_arr.append(runtime)
        num_dimensions.append(columns)

    fig, ax = plt.subplots()
    ax.set_title("Runtime versus Number of Dimensions")
    ax.set_xlabel("Number of Dimensions")
    ax.set_ylabel("Runtime (Max of 5 Iterations)")
    ax.plot(num_dimensions, runtime_arr, color='r')
    plt.show()


def plot_runtime_versus_num_transactions():
    files = ['../Data/shuttle/train-5.arff','../Data/shuttle/train-10.arff',
             '../Data/shuttle/train-15.arff','../Data/shuttle/train-20.arff',
             '../Data/shuttle/train-25.arff','../Data/shuttle/train-30.arff','../Data/shuttle/train-35.arff','../Data/shuttle/train-40.arff']
    values_of_k = 7
    runtime_arr = []
    num_transactions = []

    for idx,file in enumerate(files):
        dataframe, classes, original_dataframe = parse_arff_file(file)
        headers = list()
        full_data_averages = list()
        for col in list(dataframe):
            headers.append(col)
            full_data_averages.append(dataframe[col].mean())

        rows, columns = dataframe.shape
        clusters, original_centroids, final_centroids, num_iterations, runtime, error = kmeans.k_means_clustering(dataframe, values_of_k, 5, 0.01)
        runtime_arr.append(runtime)
        num_transactions.append(rows)

    fig, ax = plt.subplots()
    ax.set_title("Runtime versus Number of Instances")
    ax.set_xlabel("Number of Instances")
    ax.set_ylabel("Runtime (Max of 5 Iterations)")
    ax.plot(num_transactions, runtime_arr, color='r')
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


def print_k_means_data(original_dataframe, clusters, original_centroids, final_centroids, num_iterations, runtime, error, classes, normalize):
    print("\nkMeans")
    print("======")
    print("\nNumber of iterations: {}".format(num_iterations))
    print("Within cluster sum of squared errors: {}".format(round(error, 3)))
    print("\nInitial starting points (random):")

    rows, columns = original_dataframe.shape
    headers = list()
    full_data_averages = list()
    for col in list(original_dataframe):
        headers.append(col)
        full_data_averages.append(original_dataframe[col].mean())

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

        # if current_attribute.lower() == 'class':
        #     prediction = int(round(full_data_averages[i], 0))

        #     if classes is not None:
        #         row_str = current_attribute + "\t" + str(classes[prediction]) + "\t"
        #         for j in range(k+1):
        #             prediction_idx = int(round(final_centroids[j][0][i], 0))
        #             row_str += "\t" + str(classes[prediction_idx])
        #     else:
        #         row_str = current_attribute + "\t" + str(prediction) + "\t"
        #         for j in range(k+1):
        #             prediction_idx = int(round(final_centroids[j][0][i], 0))
        #             row_str += "\t" + str(prediction_idx)
        # else:
        row_str = current_attribute + "\t" + str(round(full_data_averages[i], 3)) + "\t"
        for j in range(k+1):
            row_str += "\t" + str(round(final_centroids[j][0][i], 3))
        print(row_str)

    print("\nTime taken to build model (full training data) : {} seconds".format(round(runtime, 5)))
    
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
            x_plt.append(instance[1][0])
            y_plt.append(instance[1][1])
            z_plt.append(instance[1][2])
        print("Added {} instances to cluster {} to print in color {}.".format(len(x_plt),k,colors[k]))
        ax.scatter(x_plt, y_plt, z_plt, color=colors[k])

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])

    ax.legend()
    plt.show()


def parse_arff_file(filename, normalize):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    classes = []
    with open(filename) as file:
        df = a2p.load(file)

        try:
            df.iloc[:,-1] = df.iloc[:,-1].apply(int)
            classes = None
        except:
            le = preprocessing.LabelEncoder()
            le.fit(df.iloc[:,-1])
            classes = list(le.classes_)
            df.iloc[:,-1] = le.transform(df.iloc[:,-1]) 

        df = df.select_dtypes(include=numerics)

        original_dataframe = copy.deepcopy(df)
        if normalize:
            headers = df.columns
            x = df.values
            scaler = preprocessing.Normalizer()
            scaled_df = scaler.fit_transform(df)
            df = pd.DataFrame(scaled_df)
            df.columns=headers

        return df, classes, original_dataframe


def parse_command_line_args(args):
    filename = ''
    k = 0
    epsilon = 0
    max_iterations = 0
    seed = 10
    normalize = False

    if len(args) > 12:
        print("Incorrect number of CLI arguments. Please see README for an example command for how to run this program.")
        sys.exit()
    else:
        if '-f' in args:
            idx = args.index('-f')

            if isinstance(args[idx+1], str):
                input_filename = args[idx+1]

                user_file = Path(input_filename)
                if not user_file.exists() or not user_file.is_file():
                    print("Filename: {} does not exist. Exiting...".format(user_file))
                    sys.exit()

                filename = input_filename
                print("Using the specified value for input filename: " + str(input_filename))
            else:
                print("Incorrect paramter specification. Exiting...")
                sys.exit()

        if '-k' in args:
            idx = args.index('-k')

            if isinstance(args[idx+1], str):
                try:
                    k = int(args[idx+1])
                    print("Using the specified value for k: " + str(k) + ".")
                except:
                    print("Could not parse {} into an integer.".format(args[idx+1]))
            else:
                print("Incorrect paramter specification. Exiting...")
                sys.exit()

        if '-e' in args:
            idx = args.index('-e')

            if isinstance(args[idx+1], str):
                try:
                    epsilon = float(args[idx+1])
                    print("Using the specified value for epsilon: " + str(epsilon) + ".")
                except:
                    print("Could not parse {} into a float.".format(args[idx+1]))
            else:
                print("Incorrect paramter specification. Exiting...")
                sys.exit()

        if '-i' in args:
            idx = args.index('-i')

            if isinstance(args[idx+1], str):
                try:
                    max_iterations = int(args[idx+1])
                    print("Using the specified value for max iterations: " + str(max_iterations) + ".")
                except:
                    print("Could not parse {} into an integer.".format(args[idx+1]))
            else:
                print("Incorrect paramter specification. Exiting...")
                sys.exit()
        if '-s' in args:
            idx = args.index('-s')

            if isinstance(args[idx+1], str):
                try:
                    seed = int(args[idx+1])
                    print("Using the specified value for seed: " + str(seed) + ".")
                except:
                    print("Could not parse {} into an integer.".format(args[idx+1]))
            else:
                print("Incorrect paramter specification. Exiting...")
                sys.exit()
        else:
            print("Using default value for seed: {}".format(seed))

        if '-n' in args:
            print("Program will normalize data before running k-means.")
            normalize = True
        else:
            print("Program will not normalize data before running k-means.")

        return filename, k, epsilon, max_iterations, seed, normalize


if __name__ == '__main__':
    main()