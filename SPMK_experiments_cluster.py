# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:01:35 2016

@author: jonyoung
"""

# import the stuff we need
import pandas as pd
import os
#import SPMK_3d
import numpy as np
from sklearn import svm, cross_validation, metrics


# parameters
n_intensity_levels = 8
n_histogram_levels = 7

def mccv(data, labels, classifier, n_iter, test_size):
    
    n = len(data)
    accs = []
    
    splits = cross_validation.ShuffleSplit(n, n_iter, test_size)
    for train_index, test_index in splits :
        
        K_train = data[train_index,:][:, train_index]
        K_test = data[test_index, :][:, train_index]
        labels_train = labels[train_index]
        labels_test = labels[test_index]
        classifier.fit(K_train, labels_train)
        accs.append(classifier.score(K_test, labels_test))
        
    return accs
    
def repeated_k_cv(data, labels, classifier, repeats, k) :
    
    n = len(data)
    accs = np.zeros((repeats,))
    for i in range(repeats) :
        
        # generate a permutation of the subjects        
        perm = np.random.permutation(range(n))
        
        # shuffle the labels and kernel matrix according to the permutation
        rep_labels = labels[perm]
        rep_K = data[perm, :][:, perm]
        score = cross_validation.cross_val_score(classifier, rep_K, rep_labels - 1, cv=k)
        accs[i] = np.mean(score)
    
    return accs
        
        

# directories
data_dir = '~/Data/IXI/GM-masked-groupwise-sub1pt5/'
metadata_dir = '~/Data/IXI/'

# read through images
files = os.listdir(data_dir)

# extract IXI id number from filenames
file_IXI_ids = map(lambda x: int(x.split('-')[0][3:]), files)

# load metadata0.2
IXI_metadata = pd.read_csv(metadata_dir + 'IXI.csv')

# find duplicates and remove them
IXI_metadata.drop_duplicates(['IXI_ID'], keep=False, inplace=True)

# drop rows where there is no image listed ie DATE_AVAILABLE=0
IXI_metadata = IXI_metadata[IXI_metadata['DATE_AVAILABLE'] != 0]

# drop rows where there is no age value
IXI_metadata = IXI_metadata[IXI_metadata['AGE'].notnull()]

# drop rows where there is no image file corresponding with the IXI_ID
IXI_metadata = IXI_metadata[IXI_metadata['IXI_ID'].isin(file_IXI_ids)]

IXI_metadata = IXI_metadata.reindex(np.random.permutation(IXI_metadata.index))

# create a data structure to hold grid search results
grid_search_results = np.zeros((5, 7))

clf = svm.SVC(kernel='precomputed')

# loop through possible parameters
#for i in range(1, 6) :
#    
#    for j in range(1, 8) :
#        
#        n_intensity_levels = 2 ** i
#        n_histogram_levels = j
#        print 'n_intensity_levels = ' + str(n_intensity_levels)
#        print 'n_histogram_levels = ' + str(n_histogram_levels)
#        if (n_intensity_levels == 16 and n_histogram_levels == 7) or n_intensity_levels == 32 :
#            K = np.genfromtxt(metadata_dir + 'kernels/SPMK_intersection_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '_3.csv', delimiter=',')
#        else :
#            K = np.genfromtxt(metadata_dir + 'kernels/SPMK_intersection_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', delimiter=',')
#        accs = repeated_k_cv(K, np.array(IXI_metadata['SEX_ID'].tolist()), clf, 20, 20)
#        np.mean(accs)
#        grid_search_results[i-1, j-1] = np.mean(accs)
#
#        print 'mean accuracy = ' + str(np.mean(accs))
#        
#best_params = np.unravel_index(grid_search_results.argmax(), grid_search_results.shape)
#print grid_search_results
#print best_params

results = np.zeros((10,))
selected_parameters = np.zeros((10, 2))
selected_kernel_type = []

# proper nested grid 10-fold cv with inner loop to select parameters
kf = cross_validation.KFold(len(IXI_metadata), 10)
ind = 0 
for train_index, test_index in kf:
    
    print 'ind = ' + str(ind)
    
    # do a grid search on the training group
    
    # loop through possible parameters    
    intersection_results = np.zeros((5, 7))
    bhattacharyya_results = np.zeros((5, 7))       
    
    for i in range(1, 6) :
    
        for j in range(1, 8) :
        
            n_intensity_levels = 2 ** i
            n_histogram_levels = j
            print 'n_intensity_levels = ' + str(n_intensity_levels)
            print 'n_histogram_levels = ' + str(n_histogram_levels)
            
            # first try intersection kernel            
            if (n_intensity_levels == 16 and n_histogram_levels == 7) or n_intensity_levels == 32 :
                K = np.genfromtxt(metadata_dir + 'kernels/SPMK_intersection_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '_3.csv', delimiter=',')
            else :
                K = np.genfromtxt(metadata_dir + 'kernels/SPMK_intersection_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', delimiter=',')
            K = K[train_index, :][:, train_index]
            labels = np.array(IXI_metadata['SEX_ID'].tolist())[train_index]
            accs = repeated_k_cv(K, labels, clf, 20, 20)
            intersection_results[i-1, j-1] = np.mean(accs)
            
            # next try bhattacharyya kernel
            if (n_intensity_levels == 16 and n_histogram_levels == 7) or n_intensity_levels == 32 :
                K = np.genfromtxt(metadata_dir + 'kernels/SPMK_bhattacharyya_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '_3.csv', delimiter=',')
            else :
                K = np.genfromtxt(metadata_dir + 'kernels/SPMK_bhattacharyya_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', delimiter=',')
            K = K[train_index, :][:, train_index]
            labels = np.array(IXI_metadata['SEX_ID'].tolist())[train_index]
            accs = repeated_k_cv(K, labels, clf, 20, 20)
            bhattacharyya_results[i-1, j-1] = np.mean(accs)
            
    print 'bhattacharyya results:'
    print bhattacharyya_results
    
    print 'intersection results:'
    print intersection_results
            
    # find most accurate set of parameters
    intersection_best_parameters = np.unravel_index(intersection_results.argmax(), intersection_results.shape)
    
    # find most accurate results
    intersection_best_result = intersection_results[intersection_best_parameters]
                
    # find most accurate set of parameters
    bhattacharyya_best_parameters = np.unravel_index(bhattacharyya_results.argmax(), bhattacharyya_results.shape)
    
    # find most accurate results
    bhattacharyya_best_result = bhattacharyya_results[bhattacharyya_best_parameters]
    
    # compare the intersection and bhattacharyya results
    if intersection_best_result > bhattacharyya_best_result :
        
        selected_parameters[ind, :] = intersection_best_parameters
        selected_kernel_type.append('intersection')
        print 'best kernel type is intersection' 
        
    else :
        
        selected_parameters[ind, :] = bhattacharyya_best_parameters
        selected_kernel_type.append('bhattacharyya')
        print 'best kernel type is bhattacharyya'
    
    # train on training data with the best parameters/kernel and test on the left over data
    best_n_intensity_levels = 2 ** selected_parameters[ind, 0]
    best_n_histogram_levels = selected_parameters[ind, 1]
    print 'selected_parameters = ' + selected_parameters[i, :]
    print 'best number of intensity levels is ' + str(best_n_intensity_levels)    
    print 'best number of histogram levels is ' + str(best_n_histogram_levels)     
    K = np.genfromtxt(metadata_dir + 'kernels/SPMK_' + selected_kernel_type[ind] + '_' + str(int(best_n_intensity_levels)) + '_' + str(int(best_n_histogram_levels)) + '.csv', delimiter=',')
    K_train = K[train_index, :][:, train_index]
    K_test = K[test_index, :][:, train_index]
    train_labels = np.array(IXI_metadata['SEX_ID'].tolist())[train_index]
    test_labels = np.array(IXI_metadata['SEX_ID'].tolist())[test_index]
    clf.fit(K_train, train_labels)
    preds = clf.predict(K_test)
    results[ind] = metrics.accuracy_score(test_labels, preds)
    print 'accuracy = ' + str(metrics.accuracy_score(test_labels, preds))
    
    ind += 1
    
print selected_parameters
print selected_kernel_type              
print results
print np.mean(results) 