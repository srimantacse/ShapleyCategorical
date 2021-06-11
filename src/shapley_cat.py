import math
import random
import numpy as np
from numpy import array
from utils import list_max, list_min, compute_dist, inverseNormDist
import time

def f(dist, dMax):
	return 1 -(dist/(dMax + 1))
	
def shapley(data, P=5):
	rowCount   = data.shape[0]
	dMax       = 0.0

	print("\n>>================= Shapley ================================")
	print("Info:: Shapley running with P: " + str(P) + " Data Shape: " + str(data.shape)) 
	shapley  = [0] * rowCount

	random.seed(time.time())
	randomDataIndex = random.sample(range(0, rowCount), P)

	# Find dMax, distance
	print("Info:: Distnace Calculation Started...")

	selectedData = [data[x] for x in randomDataIndex]
	distance     = compute_dist(data, selectedData, True)
	dMax         = np.amax(distance)
	print("Info:: Calculatd Distance Shape: " + str(array(distance).shape))
	print("Info:: Calculatd Distance Max: "   + "{:0,.2f}".format(dMax))	

	# Find Shapley
	print("Info:: Shapley Calculation Started...")
	for i in range(rowCount):
		result = 0.0
		for j in range(len(randomDataIndex)):
			result = result + inverseNormDist(distance[i][j], dMax)
			
		shapley[i] = (result / 2.0)		
		
	print("Info:: Shapley Calculation Done.")
	print("-------------------------------------------------------------")

	return list(shapley), dMax

def shapley_cat(data, delta=.9, P=5):
	rowCount   = data.shape[0]
	dMax       = 0.0
	
	print("\n=================== ShapleyCat ===================================")
	print("Info:: ShapleyCat running with P: " + str(P) + " Data Shape: " + str(data.shape) + "  Delta: " + str(delta)) 
	_shapley, dMax  = shapley(data, P)
	
	# Find Cluster
	itr        = 0
	validIndex = [i for i in range(rowCount)]
	
	K          = []
	KIndex     = []
	Fi         = array(_shapley)
	Q          = data.tolist()
	
	print("Info:: Clustering Started...")
	while True:
		itr = itr + 1
		
		x = array(validIndex)
		validList = x[x != -1]
		
		print("--Iteration:\t" + str(itr) + "\tValidIndex No: " + str(len(validList)))
		
		if len(validList) == 0:
			break
		
		t, v = list_max(Fi)
		xt   = Q[t]
		
		# K = K U {xt}
		K.append(xt)
		KIndex.append(t)
		
		#print ("t: " + str(t)+ " K: " + str(K)+"\n")		
		for i in range(len(Q)):
			if Fi[i] == 0:
				continue
			xi = Q[i]
			dist = compute_dist(array(xi).reshape(1, -1), array(xt).reshape(1, -1), True)
			calc_dist = sum(f(dist, dMax))
			#print(calc_dist)
			if calc_dist >= delta:
				validIndex[i] = -1
				Q[i]          = []
				Fi[i]         = 0		
	
	print("Info:: ShapleyCat Cluster Length:\t" + str(len(list(K))))
	print("=================== ShapleyCat ===================================\n")
	return list(K), list(KIndex)