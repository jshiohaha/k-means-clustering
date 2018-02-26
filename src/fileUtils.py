import sys

from arff2pandas import a2p

def parse_arff_file(filename):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    with open(filename) as file:
        df = a2p.load(file)
        new_df = df.select_dtypes(include=numerics)
        return new_df