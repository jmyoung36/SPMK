# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:37:51 2016

@author: jonyoung
"""

# import the stuff we need
import pandas as pd
import os
#import SPMK_3d
import numpy as np
from sklearn import svm, kernel_ridge, cross_validation, metrics


# parameters
n_intensity_levels = 8
n_histogram_levels = 7

def mccv(data, labels, classifier, n_iter, test_size):
    
    n = len(data)
    accs = []
    
    i = 0
    
    splits = cross_validation.ShuffleSplit(n, n_iter, test_size)
    for train_index, test_index in splits :
        
        #print i
        
        K_train = data[train_index,:][:, train_index]
        K_test = data[test_index, :][:, train_index]
        labels_train = labels[train_index]
        labels_test = labels[test_index]
        classifier.fit(K_train, labels_train)
        accs.append(classifier.score(K_test, labels_test))
        
        i+=1
        
    return accs
    
def repeated_k_cv(data, labels, classifier, repeats, k, metric) :
    
    n = len(data)
    accs = np.zeros((repeats,))
    for i in range(repeats) :
        
        #print i        
        
        # generate a permutation of the subjects        
        perm = np.random.permutation(range(n))
        
        # shuffle the labels and kernel matrix according to the permutation
        rep_labels = labels[perm]
        rep_K = data[perm, :][:, perm]
        score = cross_validation.cross_val_score(classifier, rep_K, rep_labels - 1, cv=k, scoring=metric)
        accs[i] = np.mean(score)
    
    return accs
        
# directories
data_dir = '/home/jonyoung/IoP_data/Data/IXI/GM-masked-groupwise-mod/'
metadata_dir = '/home/jonyoung/IoP_data/Data/IXI/'
kernel_dir = '/home/jonyoung/IoP_data/Data/IXI/kernels/original/'

K_lin = np.genfromtxt(kernel_dir + 'K_lin.csv', delimiter=',')

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

perm = np.random.permutation(len(IXI_metadata))
#IXI_metadata = IXI_metadata.iloc[perm]

# create a data structure to hold grid search results
grid_search_results = np.zeros((5, 7))

############################################################################
############### CLASSIFICATION - GENDER PREDICTION #########################
############################################################################

clf = svm.SVC(kernel='precomputed') 

n_folds = 10

lin_results = np.zeros((len(IXI_metadata), ))
SPMK_results = np.zeros((len(IXI_metadata),))


lin_fold_results = np.zeros((n_folds,))
SPMK_fold_results = np.zeros((n_folds,))
selected_parameters = np.zeros((n_folds, 2))
selected_kernel_type = []
mean_intersection_results = np.zeros((5, 7))
mean_bhattacharyya_results = np.zeros((5, 7))

# proper nested grid 10-fold cv with inner loop to select parameters
kf = cross_validation.KFold(len(IXI_metadata), n_folds) 
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
            if os.path.isfile(kernel_dir + 'SPMK_intersection_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv') :
                K = np.genfromtxt(kernel_dir + 'SPMK_intersection_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', delimiter=',')
                K = K[train_index, :][:, train_index]
                labels = np.array(IXI_metadata['SEX_ID'].tolist())[train_index]
                acc = np.mean(repeated_k_cv(K, labels, clf, 50, 10, 'accuracy'))
                intersection_results[i-1, j-1] = acc
            
            # next try bhattacharyya kernel
            if os.path.isfile(kernel_dir + 'SPMK_bhattacharyya_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv') :
                K = np.genfromtxt(kernel_dir + 'SPMK_bhattacharyya_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', delimiter=',')
                K = K[train_index, :][:, train_index]
                acc = np.mean(repeated_k_cv(K, labels, clf, 50, 20, 'accuracy'))
                bhattacharyya_results[i-1, j-1] = acc
            
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
    train_labels = np.array(IXI_metadata['SEX_ID'].tolist())[train_index]
    test_labels = np.array(IXI_metadata['SEX_ID'].tolist())[test_index]
    if intersection_best_result > bhattacharyya_best_result :
        
        selected_parameters[ind, :] = intersection_best_parameters
        selected_kernel_type.append('intersection')
        print 'best kernel type is intersection' 
        
    else :
        
        selected_parameters[ind, :] = bhattacharyya_best_parameters
        selected_kernel_type.append('bhattacharyya')
        print 'best kernel type is bhattacharyya'
    
    # store the scores across all parameters for the fold
    mean_intersection_results = mean_intersection_results + intersection_results    
    mean_bhattacharyya_results = mean_bhattacharyya_results + bhattacharyya_results
    
    # train on training data with the best parameters/kernel and test on the left over data
    best_n_intensity_levels = 2 ** (selected_parameters[ind, 0] + 1)
    best_n_histogram_levels = selected_parameters[ind, 1] + 1
    print 'selected_parameters = ' + str(selected_parameters[ind, :])
    print 'best number of intensity levels is ' + str(best_n_intensity_levels)    
    print 'best number of histogram levels is ' + str(best_n_histogram_levels)     
    K = np.genfromtxt(kernel_dir + 'SPMK_' + selected_kernel_type[ind] + '_' + str(int(best_n_intensity_levels)) + '_' + str(int(best_n_histogram_levels)) + '.csv', delimiter=',')
    K_train = K[train_index, :][:, train_index]
    K_test = K[test_index, :][:, train_index]
    train_labels = np.array(IXI_metadata['SEX_ID'].tolist())[train_index]
    test_labels = np.array(IXI_metadata['SEX_ID'].tolist())[test_index]
    clf.fit(K_train, train_labels)
    preds = clf.predict(K_test)
    SPMK_fold_result = metrics.accuracy_score(test_labels, preds)
    SPMK_fold_results[ind] = SPMK_fold_result
    SPMK_results[test_index] = preds
    print 'SPMK accuracy for this fold = ' + str(SPMK_fold_result)
    
    # train and test on the linear kernel and store result for the same split
    K_train = K_lin[train_index, :][:, train_index]
    K_test = K_lin[test_index, :][:, train_index]
    clf.fit(K_train, train_labels)
    preds = clf.predict(K_test)
    lin_fold_result = metrics.accuracy_score(test_labels, preds)
    lin_fold_results[ind] = lin_fold_result
    lin_results[test_index] = preds
    print 'linear kernel accuracy for this fold = ' + str(lin_fold_result)
     
    ind += 1


print 'SPMK results:'    
print selected_parameters
print selected_kernel_type              
print np.mean(SPMK_fold_results)
print metrics.accuracy_score(IXI_metadata['SEX_ID'].tolist(), SPMK_results)

print 'SPMK grid search results:'
print mean_intersection_results/n_folds
print mean_bhattacharyya_results/n_folds

print 'linear results:'  
print np.mean(lin_fold_results)
print metrics.accuracy_score(IXI_metadata['SEX_ID'].tolist(), lin_results)

############################################################################
###################### REGRESSION - AGE PREDICTION #########################
############################################################################
        
np.set_printoptions(precision=3)

reg = kernel_ridge.KernelRidge(kernel='precomputed')

lin_results = np.zeros((len(IXI_metadata), ))
SPMK_results = np.zeros((len(IXI_metadata),))

n_folds = 10

lin_fold_results = np.zeros((n_folds,))
SPMK_fold_results = np.zeros((n_folds,))
selected_parameters = np.zeros((n_folds, 2))
selected_kernel_type = []
mean_intersection_results = np.zeros((5, 7))
mean_bhattacharyya_results = np.zeros((5, 7))

# proper nested grid 10-fold cv with inner loop to select parameters
kf = cross_validation.KFold(len(IXI_metadata), n_folds) 
ind = 0 
for train_index, test_index in kf:
    
    print 'ind = ' + str(ind)
#    
#    # do a grid search on the training group
#    
#    # loop through possible parameters    
#    intersection_results = np.zeros((5, 7)) - 999
#    bhattacharyya_results = np.zeros((5, 7)) - 999    
#    
#    for i in range(1, 6) :
#    
#        for j in range(1, 8) :
#        
#            n_intensity_levels = 2 ** i
#            n_histogram_levels = j
#            #print 'n_intensity_levels = ' + str(n_intensity_levels)
#            #print 'n_histogram_levels = ' + str(n_histogram_levels)
#            
#            # first try intersection kernel            
#            if os.path.isfile(kernel_dir + 'SPMK_intersection_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv') :
#                K = np.genfromtxt(kernel_dir + 'SPMK_intersection_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', delimiter=',')
#                K = K[train_index, :][:, train_index]
#                labels = np.array(IXI_metadata['AGE'].tolist())[train_index]
#                #maes = repeated_k_cv(K, labels, reg, 20, 20, 'mean_absolute_error')hao atoll
#                #intersection_results[i-1, j-1] = np.mean(maes)
#                rmses = np.sqrt(-1 * repeated_k_cv(K, labels, reg, 20, 20, 'mean_squared_error'))
#                intersection_results[i-1, j-1] = np.mean(rmses) * -1
#            100
#            # next try bhattacharyya kernel
#            if os.path.isfile(kernel_dir + 'SPMK_bhattacharyya_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv') :
#                K = np.genfromtxt(kernel_dir + 'SPMK_bhattacharyya_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', delimiter=',')
#                K = K[train_index, :][:, train_index]
#                #maes = repeated_k_cv(K, labels, reg, 20, 20, 'mean_absolute_error')
#                #bhattacharyya_results[i-1, j-1] = np.mean(maes)
#                rmses = np.sqrt(-1 * repeated_k_cv(K, labels, reg, 20, 20, 'mean_squared_error'))
#                bhattacharyya_results[i-1, j-1] = np.mean(rmses) * -1
#            
#    #print 'bhattacharyya results:'
#    #print bhattacharyya_results
#    
#    #print 'intersection results:'
#    #print intersection_results
#            
#    # find most accurate set of parameters
#    intersection_best_parameters = np.unravel_index(intersection_results.argmax(), intersection_results.shape)
#    
#    # find most accurate results
#    intersection_best_result = intersection_results[intersection_best_parameters]
#                
#    # find most accurate set of parameters
#    bhattacharyya_best_parameters = np.unravel_index(bhattacharyya_results.argmax(), bhattacharyya_results.shape)
#    
#    # find most accurate results
#    bhattacharyya_best_result = bhattacharyya_results[bhattacharyya_best_parameters]
#    
#    # compare the intersection and bhattacharyya results    train_labels = np.array(IXI_metadata['SEX_ID'].tolist())[train_index]
#    test_labels = np.array(IXI_metadata['AGE'].tolist())[test_index]
#    if intersection_best_result > bhattacharyya_best_result :
#        
#        selected_parameters[ind, :] = intersection_best_parameters
#        selected_kernel_type.append('intersection')print np.sqrt(metrics.mean_squared_error(IXI_metadata['AGE'].tolist(), lin_results))
#        #print 'best kernel type is intersection' 
#        
#    else :
#        print 'SPMK results:'    
#        print selected_parameters
#        print selected_kernel_type              
#        print np.mean(SPMK_fold_results)
#        #print metrics.mean_absolute_error(IXI_metadata['AGE'].tolist(), SPMK_results)
#        print np.sqrt(metrics.mean_squared_error(IXI_metadata['AGE'].tolist(), SPMK_results))
#
#        print 'SPMK grid search results:'
#        print mean_intersection_results/n_folds
#        print mean_bhattacharyya_results/n_folds
#        
#        selected_parameters[ind, :] = bhattacharyya_best_parameters
#        selected_kernel_type.append('bhattacharyya')
#        #print 'best kernel type is bhattacharyya'
#        
#    # store the scores across all parameters for the fold
#    mean_intersection_results = mean_intersection_results + intersection_results    
#    mean_bhattacharyya_results = mean_bhattacharyya_results + bhattacharyya_results
#    
#    # train on training data with the best parameters/kernel and test on the left over data
#    best_n_intensity_levels = 2 ** (selected_parameters[ind, 0] + 1)
#    best_n_histogram_levels = selected_parameters[ind, 1] + 1
#    #print 'selected_parameters = ' + str(selected_parameters[ind, :])
#    #print 'best number of intensity levels is ' + str(best_n_intensity_levels)    
#    #print 'best number of histogram levels is ' + str(best_n_histogram_levels)     
#    K = np.genfromtxt(kernel_dir + 'SPMK_' + selected_kernel_type[ind] + '_' + str(int(best_n_intensity_levels)) + '_' + str(int(best_n_histogram_levels)) + '.csv', delimiter=',')
#    K_train = K[train_index, :][:, train_index]
#    K_test = K[test_index, :][:, train_index]
#    train_labels = np.array(IXI_metadata['AGE'].tolist())[train_index]
#    test_labels = np.array(IXI_metadata['AGE'].tolist())[test_index]
#    reg.fit(K_train, train_labels)
#    preds = reg.predict(K_test)
#    #SPMK_fold_result = metrics.mean_absolute_error(test_labels, preds)
#    SPMK_fold_result = np.sqrt(metrics.mean_squared_error(test_labels, preds))
#    SPMK_fold_results[ind] = SPMK_fold_result
#    SPMK_results[test_index] = preds
#    #print 'SPMK mean absolute error for this fold = ' + str(metrics.mean_absolute_error(test_labels, preds))
#    print 'SPMK RMSE for this fold = ' + str(SPMK_fold_result)
#    
#    # train and test on the linear kernel and store result for the same split
#    K_train = K_lin[train_index, :][:, train_index]
#    K_test = K_lin[test_index, :][:, train_index]
#    reg.fit(K_train, train_labels)
#    preds = reg.predict(K_test)
#    #lin_fold_result = metrics.mean_absolute_error(test_labels, preds)
#    lin_fold_result = np.sqrt(metrics.mean_squared_error(test_labels, preds))
#    lin_fold_results[ind] = lin_fold_result
#    lin_results[test_index] = preds
#    #print 'linear kernel mean absolute error for this fold = ' + str(metrics.mean_absolute_error(test_labels, preds))
#    print 'linear kernel RMSE for this fold = ' + str(lin_fold_result)
#     
#    ind += 1
#
#
#print 'SPMK results:'    
#print selected_parameters
#print selected_kernel_type              
#print np.mean(SPMK_fold_results)
##print metrics.mean_absolute_error(IXI_metadata['AGE'].tolist(), SPMK_results)
#print np.sqrt(metrics.mean_squared_error(IXI_metadata['AGE'].tolist(), SPMK_results))
#
#print 'SPMK grid search results:'
#print mean_intersection_results/n_folds
#print mean_bhattacharyya_results/n_folds

#print 'linear results:'  
#print np.mean(lin_fold_results)
#print metrics.mean_absolute_error(IXI_metadata['AGE'].tolist(), lin_results)
#print np.sqrt(metrics.mean_squared_error(IXI_metadata['AGE'].tolist(), lin_results))

# save individual score for plotting
#plot_results_SPMK = np.zeros((2, len(IXI_metadata)))
#plot_results_SPMK[0, :] = IXI_metadata['AGE']
#plot_results_SPMK[1, :] = SPMK_results
#
#plot_results_lin = np.zeros((2, len(IXI_metadata)))
#plot_results_lin[0, :] = IXI_metadata['AGE']
#plot_results_lin[1, :] = lin_results

#np.savetxt(metadata_dir + 'plot_results_sub1pt5_SPMK.csv', plot_results_SPMK, delimiter=',')
#np.savetxt(metadata_dir + 'plot_results_sub1pt5_lin.csv', plot_results_lin, delimiter=',')
