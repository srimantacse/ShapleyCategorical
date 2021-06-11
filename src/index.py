from numpy import array
import numpy as np
from math import*
import copy
from sklearn.metrics.cluster import adjusted_rand_score
from utils import list_max, list_min, multiIndexOf, unique, compute_dist, point_distance


"""
	Various Cluster validity indices
   
    Minimize
    ====================
	Aplha:
	Beta:
	DB:
    
    Maximize
    ====================
	ARI:
	DUNN:
	PC:  Partition Coefficient
	NPC: Normalized Partition Coefficient
"""

def alpha(cluster, data):
	# data: Data points
	# cluster: cluster centers
	distance = compute_dist(data, cluster, True)
	
	resultedClassLabel = []
	for i in range(len(data)):
		idx, val = list_min(distance[i])
		resultedClassLabel.append(idx)
		
	result = 0.0
	for i in range(len(data)):
		dist = compute_dist(array(data[i]).reshape(1,-1), array(cluster[resultedClassLabel[i]]).reshape(1,-1), True)
		result = result + (dist**2)
		
	ret = '{:0,.2f}'.format(float(result / len(data)))
	#ret = '{:0,.2f}'.format(result / (len(unique(resultedClassLabel))))
	return float(result / len(data))
	
def beta(cluster, data):
	# data: Data points
	# cluster: cluster centers
	distance = compute_dist(data, cluster, True)
	
	resultedClassLabel = []
	for i in range(len(data)):
		idx, val = list_min(distance[i])
		resultedClassLabel.append(idx)
	
	result = 0.0
	for i in unique(resultedClassLabel):
		indexList = multiIndexOf(resultedClassLabel, i)
		elementCount = len(indexList)
		if elementCount <= 1:
			continue
			
		#_data = data[indexList] 
		#_data = list(map(list(data).__getitem__, indexList))
		_data = [data[i] for i in indexList]
		_distance = compute_dist(_data, _data, True) ** 2
		result = result + (sum(sum(_distance)) / (elementCount * (elementCount - 1)))
	
	#ret = "{0:.3f}".format(float(result / (len(unique(resultedClassLabel)))))
	ret = '{:0,.2f}'.format(result / (len(unique(resultedClassLabel))))
	return result / (len(unique(resultedClassLabel)))

def ari(actualLabel, predictedLabel):
    actualLabel    = array(actualLabel).flatten()
    predictedLabel = array(predictedLabel).flatten()
    return adjusted_rand_score(actualLabel, predictedLabel)

def pc(x, u, v, m):
    c, n = u.shape
    return np.square(u).sum()/n

def npc(data, u, clusters_cent, m):
    n, c = u.shape
    return 1 - c/(c - 1)*(1 - pc(data, u, clusters_cent, m))