# K Means Clustering Implementation

`TODO:` Insert brief background of k-means and the algorithm. (what is it used for, etc.)

K-means clustering implementation. Acceptable file format is `.arff` file in order to directly compare results with Weka. The majority of testing was done on the `iris.arff` dataset. Testing on this dataset had a few advantages. Namely, there are only `4` attributes for each instance of data (transaction). This means that the algorithm was only ever dealing with 4-dimensions. This made it much easier to comprehend what was going on. Further, the dataset is easily separable on any `3-axis` combination of the `4` explanatory variables, which means that we could easily plot the visual plot to make sure that clusters were converging to where we would expect.

#### TODO's

Implement the k-means algorithm to perform clustering in a dataset.
• Your implementation will be tested on cse.unl.edu server using the command youprovided in the README file. **(15 points)**
• **In the report**, you should write a paragraph about your program design **(15 points)**• Plot the runtime of the algorithm as a function of number of clusters, number ofdimensions and size of the dataset (number of transactions). **In the report**, you should write a paragraph to summarize the observation and elaborate on it. **(10 points)**

• Plot the goodness of clustering as a function of the number of clusters and determine the optimal number of clusters. **In the report**, you should write a paragraph to summarize the observation and elaborate on it. **(15 points)**

• Compare the performance of your algorithm with that of Weka and summarize your results. **In the report**, summarize the differences (if there is any) and elaborate on it (why/how). **(10 points)**

### Program

#### Input

There are `4` main inputs for the program, which are entered via CLI arguments:

- `-f`: represents the input data file to be read parsed and read in
- `-k`: represents the number of clusters that the clustering algorithm should try to find
- `-e`: represents a threshold such that if the change in sum of the distances from cluster centers decreases below this value, the program will terminate
- `-i`: represents the number of iterations to run before terminating if the other terminating conditions are not met

#### Output

TODO --> plots or data on the centroids, clusters, etc? probably want to match weka as closely as possible here

#### Program Commands

The program can be started by running the following command that correspond to the input parameters listed above:

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