import sys
import numpy as np
import pandas as pd

from shapley_cat import shapley_cat
from index import alpha, beta, ari, npc
from utils import encode_features, pandas_to_numpy
from kmodes.kmodes import KModes
from fkmodes import FKModes
from fkmedoids import FKMedoids

if __name__ == "__main__":
    dataSet         = '../data/cat250.data'
    sepr            = '\s+'
    delta           = .32
    chosenCluster   = 10

    dataX = pd.read_csv(dataSet, sep=sepr, header=None)
    dataY = dataX.iloc[:,-1]
    dataX = dataX.iloc[:, :-1]
    row   = dataY.shape[0]
    dataY = np.array(dataY).reshape(row, 1)
        
    print("Info:: Original Data   Size: " + str(dataX.shape))
    print("Info:: Original Class  Size: " + str(dataY.shape))

    X, _ = encode_features(pandas_to_numpy(dataX))
    Y, _ = encode_features(pandas_to_numpy(dataY))
    
    
    # shapley_cat   
    cl_bigc, cl_index = shapley_cat(X, delta, dataX.shape[0])
    cl_bigc = cl_bigc[0:chosenCluster]
    K = len(cl_bigc)

    print(dataSet, delta, K)
      
    # GT-KMode
    km_gt = KModes(n_clusters = K, init = np.array(cl_bigc) , n_init = 1, verbose = 0)
    label_km_gt    = km_gt.fit_predict(X)
    clusters_km_gt = km_gt.cluster_centroids_
    clusters_km_gt, enc_map = encode_features(clusters_km_gt)
    
    print('Executed G-KMode\t...')

    ari1    = ari(Y, label_km_gt)
    alpha1	= alpha(clusters_km_gt, X)
    beta1	= beta(clusters_km_gt, X)

    # GT-FKMode
    fkm_gt = FKModes(cl_bigc, len(cl_bigc), 1, 0)
    cost3, clusters_fkm_gt, label_fkm_gt, Um_fkm_gt = fkm_gt.fit(X)
    ari2    = ari(Y, label_fkm_gt)
    alpha2  = alpha(clusters_fkm_gt, X)
    beta2   = beta (clusters_fkm_gt, X)
    npc2    = npc(X, Um_fkm_gt, clusters_fkm_gt, 2)
    
    print('Executed G-FKMode\t...')

    # GT-FKMedoid
    gt_fkmdd = FKMedoids(X, cl_index, K, 100000, m = 2)
    gt_dd_medoid, gt_dd_labels, gt_dd_Jm, gt_dd_Um = gt_fkmdd.fit()
    ari3    = ari(Y, gt_dd_labels)
    alpha3  = alpha(gt_dd_medoid, X)
    beta3   = beta (gt_dd_medoid, X)
    npc3    = npc(X, gt_dd_Um, gt_dd_medoid, 2)
    
    print('Executed G-FKMdd\t...')


    print("\nMinimize\n----------------")
    print ('{:12s} {:7s}\t{:7s}\t{:7s}'.format('', 'G-KMode', 'G-FKMode', 'G-FKMdd'))
    print ('{:12s} {:3.3f}\t{:3.3f}\t{:3.3f}'.format('Alpha', alpha1, alpha2, alpha3))
    print ('{:12s} {:3.3f}\t{:3.3f}\t{:3.3f}'.format('Beta', beta1, beta2, beta3))

    
    print("\nMaximize\n----------------")
    print ('{:12s} {:3.3f}\t{:3.3f}\t{:3.3f}'.format('ARI', ari1, ari2, ari3))
    print ('{:12s} {:3s}\t{:3.3f}\t{:3.3f}'.format('NPC', 'NA', npc2, npc3))