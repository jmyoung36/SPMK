# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:37:51 2016

@author: jonyoung
"""

# import the stuff we need
import nibabel as nib
import pandas as pd
import time
import os
#import SPMK_3d
import numpy as np
from sklearn import svm, cross_validation
import SPMK_3d

# parameters
n_intensity_levels = 8
n_histogram_levels = 7
n_subjects = 475

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

# directories
data_dir = '/home/jonyoung/IoP_data/Data/IXI/GM-masked-groupwise-sub1pt5/'
metadata_dir = '/home/jonyoung/IoP_data/Data/IXI/'

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

# calculate histogram width
histogram_width =  SPMK_3d.calculate_histogram_width(n_intensity_levels, n_histogram_levels, 0)

# initialise array to hold SPMK data
#histogram_data = np.zeros((len(IXI_metadata[:n_subjects]), 65536 * 8))
histogram_data = np.zeros((len(IXI_metadata[:n_subjects]), histogram_width))


all_data = np.zeros((len(IXI_metadata[:n_subjects]), 2890000))

# loop through metadata
for i, row in zip(range(len(IXI_metadata[:n_subjects])), IXI_metadata[:n_subjects].iterrows()) :
    
    if i % 10 == 0 :   
    
        print i
    
    IXI_ID = row[1][0]

    # find the ID in the files 
    ind = file_IXI_ids.index(IXI_ID)
    
    img = nib.load(data_dir + files[ind])
    #print i
    img_data = img.get_data()
    img_data[img_data > 1.0] = 1.0
    print np.shape(img_data)
    all_data[i,:] = img_data.flatten()
    img_codes = SPMK_3d.quantise_intensities(img_data, n_intensity_levels)
    histogram_data[i,:] =  np.array(SPMK_3d.build_histogram_at_multiple_levels_rec_3d(img_codes, n_histogram_levels, 0, n_intensity_levels, [])).flatten()
    #foo =  np.array(build_histogram_at_multiple_levels_rec_3d(img_codes, n_histogram_levels, 0, n_intensity_levels, [])).flatten()
    #print np.shape(foo)
    
print 'n_histogram_levels = ' + str(n_histogram_levels)
  
K = SPMK_3d.histogram_intersection_kernel(histogram_data, False)
#print K /(np.max(np.max(K)))
clf = svm.SVC(kernel='precomputed')
scores = cross_validation.cross_val_score(clf, K, IXI_metadata[:n_subjects]['SEX_ID'].tolist(), cv=10)
print scores
print np.mean(scores)

K = SPMK_3d.bhattacharyya_coefficient_kernel(histogram_data, False)
#print K /(np.max(np.max(K)))
clf = svm.SVC(kernel='precomputed')
scores = cross_validation.cross_val_score(clf, K, IXI_metadata[:n_subjects]['SEX_ID'].tolist(), cv=10)
print scores
print np.mean(scores)

K = np.dot(all_data, np.transpose(all_data))
#print K/(np.max(np.max(K)))
clf = svm.SVC(kernel='precomputed')
scores = cross_validation.cross_val_score(clf, K, IXI_metadata[:n_subjects]['SEX_ID'].tolist(), cv=10)
print scores
print np.mean(scores)

K = np.dot(histogram_data, np.transpose(histogram_data))
#print K/(np.max(np.max(K)))
clf = svm.SVC(kernel='precomputed')
scores = cross_validation.cross_val_score(clf, K, IXI_metadata[:n_subjects]['SEX_ID'].tolist(), cv=10)
print scores
print np.mean(scores)

scores = mccv(K, np.array(IXI_metadata[:n_subjects]['SEX_ID'].tolist()), clf, 100, 0.1)
print np.mean(scores)
##np.savetxt(metadata_dir + 'kernels/SPMK_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', K, delimiter=',')

for i in range(1, 6) :
    
    for j in range(2, 8) :
        
        n_intensity_levels = 2 ** i
        n_histogram_levels = j
        
        # calculate histogram width
        histogram_width =  SPMK_3d.calculate_histogram_width(n_intensity_levels, n_histogram_levels, 0)
        histogram_data = np.zeros((len(IXI_metadata[:n_subjects]), histogram_width))
        
        for k, row in zip(range(len(IXI_metadata[:n_subjects])), IXI_metadata[:n_subjects].iterrows()) :
    
            print k
    
            IXI_ID = row[1][0]

            # find the ID in the files 
            ind = file_IXI_ids.index(IXI_ID)
    
            img = nib.load(data_dir + files[ind])
    #print i
            img_data = img.get_data()
            img_data[img_data > 1.0] = 1.0
            all_data[i,:] = img_data.flatten()
            img_codes = SPMK_3d.quantise_intensities(img_data, n_intensity_levels)
            histogram_data[i,:] =  np.array(SPMK_3d.build_histogram_at_multiple_levels_rec_3d(img_codes, n_histogram_levels, 0, n_intensity_levels, [])).flatten()
            K = SPMK_3d.histogram_intersection_kernel(histogram_data, False)
            np.savetxt(metadata_dir + 'kernels/SPMK_intersection_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', K, delimiter=',')
            K = SPMK_3d.bhattacharyya_coefficient_kernel(histogram_data, False)
            np.savetxt(metadata_dir + 'kernels/SPMK_bhattacharyya_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', K, delimiter=',')
        

# loop through metadata
#for i, row in zip(range(len(IXI_metadata[:n_subjects])), IXI_metadata.iterrows()) :
#    
#    IXI_ID = row[1][0]
#
#    # find the ID in the files 
#    ind = file_IXI_ids.index(IXI_ID)
#    
#    img = nib.load(data_dir + files[ind])
#    print i
#    img_data = img.get_data()
#    img_data[img_data > 1.0] = 1.0
#    #img_codes = quantise_intensities(img_data, n_intensity_levels)
#    #histogram_data[i,:] =  np.array(build_histogram_at_multiple_levels_rec_3d(img_codes, n_histogram_levels, 0, n_intensity_levels, [])).flatten()
#    all_data[i,:] = img_data.flatten()

#K = histogram_intersection_kernel(histogram_data, False)
#K = np.dot(all_data, np.transpose(all_data))
#
#print np.histogram(K)
#print K
#print np.shape(K)

#np.savetxt(metadata_dir + 'kernels/SPMK_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', K, delimiter=',')

#K = np.genfromtxt(metadata_dir + 'kernels/SPMK_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', delimiter=',')



     

