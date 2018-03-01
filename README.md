# K Means Clustering Implementation

`TODO:` Insert brief background of k-means and the algorithm. (what is it used for, etc.)

K-means clustering implementation. Acceptable file format is `.arff` file in order to directly compare results with Weka. The majority of testing was done on the `iris.arff` dataset. Testing on this dataset had a few advantages. Namely, there are only `4` attributes for each instance of data (transaction). This means that the algorithm was only ever dealing with 4-dimensions. This made it much easier to comprehend what was going on. Further, the dataset is easily separable on any `3-axis` combination of the `4` explanatory variables, which means that we could easily plot the visual plot to make sure that clusters were converging to where we would expect.

### Program

#### Input

There are `4` main inputs for the program, which are entered via CLI arguments:

- `-f`: represents the input data file to be read parsed and read in
- `-k`: represents the number of clusters that the clustering algorithm should try to find
- `-e`: represents a threshold such that if the change in sum of the distances from cluster centers decreases below this value, the program will terminate
- `-i`: represents the number of iterations to run before terminating if the other terminating conditions are not met
- [Optional] `-s`: represents the value of the seed value when using `random()` to force pseudo-random functions to behave deterministically. This helps to ensure repeatability. If not specified, then the k-means clustering algorithm will use a default value of `10`.
- [Optional] `-n`: represents a boolean flag that tells the program if it should normalize the data set before running the k-means algorithm.

#### Output

The program outputs information about the clustering results from k-means. This information includes initial centroid coordinates, runtime, resulting cluster attributes, and total cluster membership.

#### Program Commands

The program can be started by running the following command that correspond to the input parameters listed above:

```python -f <input_file> -k <num_clusters> -e <epsilon> -i <max_iterations> -s 1```

Or, if you don't specify the `-s` CLI argument, which denotes the value of the seed variable, then you would use the following command:

```python -f <input_file> -k <num_clusters> -e <epsilon> -i <max_iterations>```

If your default version of Python is `Python 2.x`, you will need to specify `python3` on the command line. Otherwise, running `python` will default to `Python 3.x`.

### Implementation Assumptions

• Assume that all the attributes are continuous variables.
• Your program must allow the number of clusters (k) to be specified as input.
• Your program must allow the epsilon (change in the sum of the distances from thecluster centers) to be specified as input.
• Your program must allow the number of iterations to be specified as input.

### Terminating Conditions
The program will stop if either of the following conditions hold:

1. The number of iterations is reached
2. The change in the total sum of the squares of the distances (SSD) falls below epsilon