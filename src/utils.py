import math
import numpy as np

def compute_num_dist(data1,data2):
	n1,d = np.shape(data1)
	n2,d = np.shape(data2)

	dist = np.zeros([n1,n2])
	for i in np.arange(n1):
		for j in np.arange(n2):
			dist[i][j] = np.sqrt(np.sum((data1[i] - data2[j])**2))

	return dist

def compute_catdist(data1,data2):
	if np.array(data1).ndim < 2:
		return point_distance(data1,data2)
		
	n1,d = np.shape(data1)
	n2,d = np.shape(data2)

	dist = np.zeros([n1,n2])
	for i in np.arange(n1):
		for j in np.arange(n2):
			dist[i][j] = point_distance(data1[i], data2[j])

	return dist
	
def compute_dist(data1,data2, isItCat=False):
	if isItCat:
		return compute_catdist(data1,data2)
	else:
		#cdist(center, data)
		return compute_num_dist(data1,data2)

def point_distance(Z, X):
    """
    Calculates disssimilarity between any 2 records supplied by using Simple Matching Dissimilarity Measure algorithm.
    :param Z: Record 1
    :param X: Record 2
    :return: Dissimilarity between Z and X
    """
    m = len(Z)

    dissimlarity = 0

    for j in range(m):
        if Z[j] != X[j]:
            dissimlarity += 1

    return dissimlarity

def pandas_to_numpy(x):
    return x.values if 'pandas' in str(x.__class__) else x
    
def encode_features(X, enc_map=None):
    """Converts categorical values in each column of X to integers in the range
    [0, n_unique_values_in_column - 1], if X is not already of integer type.

    If mapping is not provided, it is calculated based on the values in X.

    Unknown values during prediction get a value of -1. np.NaNs are ignored
    during encoding, and get treated as unknowns during prediction.
    """
    if enc_map is None:
        fit = True
        # We will calculate enc_map, so initialize the list of column mappings.
        enc_map = []
    else:
        fit = False

    Xenc = np.zeros(X.shape).astype('int')
    for ii in range(X.shape[1]):
        if fit:
            col_enc = {val: jj for jj, val in enumerate(np.unique(X[:, ii]))
                       if not (isinstance(val, float) and np.isnan(val))}
            enc_map.append(col_enc)
        # Unknown categories (including np.NaNs) all get a value of -1.
        Xenc[:, ii] = np.array([enc_map[ii].get(x, -1) for x in X[:, ii]])

    return Xenc, enc_map

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 
	
def nCr(n,r):
	f = math.factorial

	if n == 1:
		return 1
	return f(n) / (f(r) * f(n-r))
	
def normDist(dist, dMax):
	return dist/(dMax + 1)
	
def inverseNormDist(dist, dMax):
	return 1 - normDist(dist, dMax)
	
def list_max(l):
	if len(np.array(l).shape) == 2:
		print("list_max: 2D List...")
		max_idx_y, max_val = list_max(l[0])
		max_idx_x = 0
		itr = 0
		for row in l:
			idx, val = list_max(row)
			if max_val < val:
				max_val = val
				max_idx_x = itr
				max_idx_y = idx
				
			itr = itr + 1
			
		return ([max_idx_x, max_idx_y], max_val)
	else:
		max_idx = np.argmax(l)
		max_val = l[max_idx]
		return (max_idx, max_val)
	
def list_min(l):
	if len(np.array(l).shape) == 2:
		print("list_min: 2D List...")
		min_idx_y, min_val = list_min(l[0])
		min_idx_x = 0
		itr = 0
		for row in l:
			idx, val = list_min(row)
			if min_val > val:
				min_val = val
				min_idx_x = itr
				min_idx_y = idx
				
			itr = itr + 1
			
		return ([min_idx_x, min_idx_y], min_val)
	else:
		min_idx = np.argmin(l)
		min_val = l[min_idx]
		return (min_idx, min_val)
	
def d(d1, d2):
	x = 0.0
	for i in range(d1.shape[0]):
		x = x + ((float(d1[i]) - float(d2[i]))**2)
	dist = math.sqrt(x)
	return dist
	
def indexOf(array, data, validIndex):
	for i in range(array.shape[0]):
		if i not in validIndex:
			continue
		if d(array.loc[i], data) == 0:
			return i
	
	print("#### Wrong index found!!!")
	return -1
	
def multiIndexOf(lst, item):
	return [i for i, x in enumerate(lst) if x == item]
	
def unique(listSet): 
    unique_list = (list(set(listSet))) 
    return unique_list		