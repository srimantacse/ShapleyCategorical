from __future__ import print_function

import logging
import sys
import os
import numpy as np
import pandas as pd
import google.cloud.logging as gLogging

from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext


client = gLogging.Client()
client.setup_logging()

def pair_dist(twoRows):
    dissimlarity = 0
    X = twoRows[0].split(',')
    Y = twoRows[1].split(',')
    logging.error("The information comes: " + str(X))

    for i in range(len(Y)):
        if Y[i] != X[i]:
            dissimlarity += 1

    return dissimlarity
    
def shapley(row):
    fullData = broadcastVar.value()
    shapley  = 0.0
    
    for i in range(len(fullData)):
        shapley = shapley + pair_dist([row, fullData[i]])
        
    return shapley / 2.0

def hostname(row):
    print("hostname: " + os.uname()[1])
    return os.uname()[1]



if __name__ == "__main__":
    logging.info("ShapleyCat: Logger Module Initialized.")
    
    pathCSVFile   = "/home/srimanta_kundu19/modifiedData_1.csv"
    noOfPartition = 3
    
    # Create Spark COntext
    sc   = SparkContext.getOrCreate()
    
    # Data Read as Numpy Array
    data = pd.read_csv(pathCSVFile, header=None)
    data = np.array(data)
    
    
    # Share the Data to All Worker Nodes
    broadcastVar = sc.broadcast(data)    
    
    # Create RDD
    rdd  = sc.parallelize(data).repartition(noOfPartition)
    print(rdd.getNumPartitions())
    print(rdd.count())

    # Mapper
    shapley = rdd.map(lambda x: shapley(x)).collect()
    print(str(shapley))
    
    # Deleting the Spark Context
    sc.stop()