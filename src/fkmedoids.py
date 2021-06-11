import numpy as np
import pandas as pd
import operator
from collections import Counter
from time import time
import kmodes as km
import random
import time
from utils import encode_features, pandas_to_numpy, compute_dist, list_min


class FKMedoids():
	def __init__(self, X, init_cluster_index = [], n_clusters = 10, max_iter = 100, m = 2):
		self.n_clusters     = n_clusters
		self.data           = pandas_to_numpy(X)
		self.medoid_index   = init_cluster_index
		self.medoid         = np.zeros((self.n_clusters, self.data.shape[1]), dtype='float')
		self.max_iter       = max_iter
		self.m              = m

		self.Jm                = 0.0        
		self.labels            = None
		self.Um                = None
		self.itr               = 0
		
	def _dissimilarity(self, setA, setB):
		return compute_dist(setA, setB, True)
		
	def fit(self):
		self._process()
		
		return self.medoid, self.labels, self.Jm, self.Um

	def accuracy(self, labels, prediction):
		labels_values = np.unique(labels)
		count = 0.0

		for key in labels_values.__iter__():
			indices = [prediction[i] for i in range(len(prediction)) if labels[i] == key]
			count += max(Counter(indices).items(), key=operator.itemgetter(1))[1]

		return round(count / len(prediction), 4) * 100
	
	def _process(self):
		if self.medoid_index == []:
			random.seed(time.time())
			self.medoid_index = random.sample(range(0, self.data.shape[0]), self.n_clusters)
			print('Info:: FKMedoid Cluster Seed: Random')
		#else:
			#print('Info:: FKMedoid Cluster Seed: Provided')
		
		self.medoid = self.data[self.medoid_index]

		while True:
			self._membership()
			self._findCentroid()
			Jm_prev = self.Jm
			self._calculateJm()
			
			self.itr += 1

			if Jm_prev == self.Jm or self.itr == self.max_iter:
				break

		self.labels = self._findLabel()
		
		#print('Info:: FKMedoid takes ' + str(self.itr) + ' iteartaions to merge')
		
	def _initCentroids(self):
		return np.array(random.sample(list(self.data), self.n_clusters))

	def _calculateJm(self):
		dataDist = self._dissimilarity(self.medoid, self.data)
		self.Jm = np.sum(np.sum((self.Um ** self.m) * dataDist))

	def _membership(self):
		Z = self.medoid
		X = self.data
		k = self.n_clusters
		n = X.shape[0]

		exponent = 1 / (self.m - 1)

		W = np.zeros((k, n), dtype='float')

		for l in range(k):
			for i in range(n):
				dli = self._dissimilarity(Z[l], X[i])
				if dli == 0:
					W[l][i] = 1
				else:
					flag = False
					sum = 0
					for h in range(k):
						dhi = self._dissimilarity(Z[h], X[i])

						if h != l and dhi == 0:
							W[l][i] = 0
							flag = True
							break

						sum += pow(dli / dhi, exponent)

					if not flag:
						W[l][i] = 1 / sum

		self.Um = W

	def _findCentroid(self):
		sVal = 0
		tVal = 0
		noOfPoints = self.data.shape[0]
		cluster2PointDist = np.zeros(noOfPoints,       dtype='float')
		modClusterIndex   = np.zeros(self.n_clusters , dtype='int')
		
		u1 = self.Um ** self.m
		dataDist = self._dissimilarity(self.data, self.data)
		for i in range(self.n_clusters):		
			for k in range(noOfPoints):
				sVal += np.sum(u1[i] * dataDist[k])
				cluster2PointDist[k] = sVal
				sVal = 0
				
			modClusterIndex[i], val = list_min(cluster2PointDist)
			cluster2PointDist.fill(0)
			
		self.medoid_index = modClusterIndex
		self.medoid       = self.data[self.medoid_index]

	def _findLabel(self):
		allotment = np.zeros(self.Um.shape[1], dtype='int')

		for i in range(self.Um.shape[1]):
			allotment[i] = np.argmax(self.Um[:, i]) + 1
		return allotment