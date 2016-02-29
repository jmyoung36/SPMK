# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:44:32 2016

@author: jonyoung
"""

# import the stuff we need
import nibabel as nib
import pandas as pd
import os
import numpy as np
import SPMK_3d
import sys

# directories
data_dir = '/home/k1511004/ADNI/groupwise/mod_gm_spm/'
metadata_dir = '/home/k1511004/Data/ADNI/Metadata/'
kernel_dir = '/home/k1511004/Data/ADNI/Kernels/'

# read through images
files = os.listdir(data_dir)

# extract ADNI RID number from filenames
file_ADNI_RIDs = map(lambda x: int(x.split('_')[2]), files)

# load metadata
ADNI_metadata = pd.read_csv(metadata_dir + 'subject_diagnoses_MRI.csv')

# filter to select wanted diagnosis, scan B-field strength etc
#ADNI_metadata = ADNI_metadata[ADNI_metadata['T'] == 3]
#ADNI_metadata = ADNI_metadata[np.logical_or(ADNI_metadata['diagnosis'] == 3, ADNI_metadata['diagnosis'] == 4)]

ADNI_RIDs = ADNI_metadata['RID'].tolist()

# run for a fixed number of intensity levels
n_intensity_levels = 32

# number of histogram levels comes from the command line args
n_histogram_levels = sys.argv[1]
        
print 'n_intensity_levels = ' + str(n_intensity_levels)
print 'n_histogram_levels = ' + str(n_histogram_levels)
print 'Generating histogram pyramid...'

# calculate histogram width
histogram_width =  SPMK_3d.calculate_histogram_width(n_intensity_levels, n_histogram_levels, 0)
            
print 'Generating kernel...'
        
################################################################################################# 
#### IF WE READ EVERYTHING IN AND CALCULATE THE KERNEL IN ONE GO - FAST BUT MEMORY INTENSIVE ####
#################################################################################################
#        step_size = 40
        
#histogram_data = np.zeros((len(ADNI_metadata), histogram_width))
#print np.shape(histogram_data)
#K_histogram_intersection = np.zeros((len(ADNI_metadata), len(ADNI_metadata)))
#K_bhattacharyya = np.zeros((len(ADNI_metadata), len(ADNI_metadata)))
#K_SPMK_lin = np.zeros((len(ADNI_metadata), len(ADNI_metadata)))
#        
#for i, ADNI_RID in zip(range(len(ADNI_metadata)), ADNI_RIDs) :
#
#    if i % 10 == 0 :   
#    
#        print i
#    
#    # find the ID in the files 
#    ind = file_ADNI_RIDs.index(ADNI_RID)
#    
#    img = nib.load(data_dir + files[ind])
#    img_data = img.get_data()
#    img_data[img_data > 1.0] = 1.0
#    img_data[img_data < 0.0] = 0.0
#    histogram_data[i, :] = np.array(SPMK_3d.build_histogram_at_multiple_levels_rec_3d(SPMK_3d.quantise_intensities(img_data, n_intensity_levels), n_histogram_levels, 0, n_intensity_levels, [])).flatten()
#        
#K = SPMK_3d.histogram_intersection_kernel(histogram_data, False)
#np.savetxt(kernel_dir + 'SPMK_intersection_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '_test_90_1.csv', K, delimiter=',')
#K = SPMK_3d.bhattacharyya_coefficient_kernel(histogram_data, False)
#np.savetxt(kernel_dir + 'SPMK_bhattacharyya_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '_test_90_1.csv', K, delimiter=',')
#        
### check this!!
#K = np.dot(histogram_data, np.transpose(histogram_data))
#np.savetxt(kernel_dir + 'SPMK_lin_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '_test_90_1.csv', K, delimiter=',')
        
################################################################################ 
#### IF WE BUILD KERNEL IN MINIBATCHES - COMPROMISE IN TIME/MEMORY TRADEOFF ####
################################################################################    
                                                   
step_size = 40
        
K_histogram_intersection = np.zeros((len(ADNI_metadata), len(ADNI_metadata)))
K_bhattacharyya = np.zeros((len(ADNI_metadata), len(ADNI_metadata)))
K_SPMK_lin = np.zeros((len(ADNI_metadata), len(ADNI_metadata))) 
        
for i in range((len(ADNI_metadata)/step_size) + 1) :
            
    start_ind_1 = i * step_size
    stop_ind_1 = min((start_ind_1 + step_size, len(ADNI_metadata)))
                                    
    print 'i = ' + str(i)
    print start_ind_1, stop_ind_1   
            
    histogram_data_1 = np.zeros(((int(stop_ind_1 - start_ind_1)), histogram_width))
            
    for k in range(start_ind_1, stop_ind_1) :

        if k % 10 == 0 :   
    
            print k
    
        # find the ID in the files
        ADNI_RID = ADNI_RIDs[k]
        ind = file_ADNI_RIDs.index(ADNI_RID)
    
        img = nib.load(data_dir + files[ind])
        img_data = img.get_data()
        img_data[img_data > 1.0] = 1.0
        img_data[img_data < 0.0] = 0.0
        histogram_data_1[k - start_ind_1, :] = np.array(SPMK_3d.build_histogram_at_multiple_levels_rec_3d(SPMK_3d.quantise_intensities(img_data, n_intensity_levels), n_histogram_levels, 0, n_intensity_levels, [])).flatten()
            
    for j in range((len(ADNI_metadata)/step_size) + 1) :
                
        start_ind_2 = j * step_size
        stop_ind_2 = min((start_ind_2 + step_size, len(ADNI_metadata)))
                
        print 'j = ' + str(j)
        print start_ind_2, stop_ind_2
                
        if i == j :
                    
            K_SPMK_lin[start_ind_1:stop_ind_1, start_ind_1:stop_ind_1] = np.dot(histogram_data_1, np.transpose(histogram_data_1))
            K_histogram_intersection[start_ind_1:stop_ind_1, start_ind_1:stop_ind_1] = SPMK_3d.histogram_intersection_kernel_2(histogram_data_1, histogram_data_1, False)
            K_bhattacharyya[start_ind_1:stop_ind_1, start_ind_1:stop_ind_1] = SPMK_3d.bhattacharyya_coefficient_kernel_2(histogram_data_1, histogram_data_1, False)
                    
        else :
                    
            histogram_data_2 = np.zeros((int(stop_ind_2 - start_ind_2), histogram_width))
            for k in range(start_ind_2, stop_ind_2) :

                if k % 10 == 0 :   
    
                    print k
    
                # find the ID in the files
                ADNI_RID = ADNI_RIDs[k]
                ind = file_ADNI_RIDs.index(ADNI_RID)
    
                img = nib.load(data_dir + files[ind])
                img_data = img.get_data()
                img_data[img_data > 1.0] = 1.0
                img_data[img_data < 0.0] = 0.0
                histogram_data_2[k - start_ind_2, :] = np.array(SPMK_3d.build_histogram_at_multiple_levels_rec_3d(SPMK_3d.quantise_intensities(img_data, n_intensity_levels), n_histogram_levels, 0, n_intensity_levels, [])).flatten()
                    
            K_SPMK_lin[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = np.dot(histogram_data_1, np.transpose(histogram_data_2))
            K_histogram_intersection[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = SPMK_3d.histogram_intersection_kernel_2(histogram_data_1, histogram_data_2, False)
            K_bhattacharyya[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = SPMK_3d.bhattacharyya_coefficient_kernel_2(histogram_data_1, histogram_data_2, False)
    
np.savetxt(kernel_dir + 'SPMK_intersection_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', K_histogram_intersection, delimiter=',')
np.savetxt(kernel_dir + 'SPMK_bhattacharyya_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', K_bhattacharyya, delimiter=',')
np.savetxt(kernel_dir + 'SPMK_lin_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', K_SPMK_lin, delimiter=',')