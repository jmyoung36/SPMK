# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:16:43 2016

@author: jonyoung
"""

# import the stuff we need
import nibabel as nib
import pandas as pd
import os
import numpy as np
import SPMK_3d
from sklearn import svm, cross_validation

# directories
data_dir = '/home/jonyoung/Local_data _to_back_up/ADNI/groupwise/mod_gm_spm/'
metadata_dir = '/home/jonyoung/IoP_data/Data/ADNI/Metadata/'
kernel_dir = '/home/jonyoung/IoP_data/Data/ADNI/Kernels/'

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

for i in range(4, 6) :
    
    if i == 4 :
        
        j_range = range(7, 8)
        
    else :
        
        j_range = range(1, 8)
    
    for j in j_range :
        
        n_intensity_levels = 2 ** i
        n_histogram_levels = j
        
        print 'n_intensity_levels = ' + str(n_intensity_levels)
        print 'n_histogram_levels = ' + str(n_histogram_levels)
        print 'Generating histogram pyramid...'

        # calculate histogram width
        histogram_width =  SPMK_3d.calculate_histogram_width(n_intensity_levels, n_histogram_levels, 0)
            
        print 'Generating kernels...'
        
        ################################################################################################# 
        #### IF WE READ EVERYTHING IN AND CALCULATE THE KERNEL IN ONE GO - FAST BUT MEMORY INTENSIVE ####
        #################################################################################################
        #        step_size = 40
        
#        histogram_data = np.zeros((len(ADNI_metadata), histogram_width))
#        print np.shape(histogram_data)
#        K_histogram_intersection = np.zeros((len(ADNI_metadata), len(ADNI_metadata)))
#        K_bhattacharyya = np.zeros((len(ADNI_metadata), len(ADNI_metadata)))
#        K_SPMK_lin = np.zeros((len(ADNI_metadata), len(ADNI_metadata)))
#        
#        for k, ADNI_RID in zip(range(len(ADNI_metadata)), ADNI_RIDs) :
#
#            if k % 10 == 0 :   
#    
#                print k
#    
#            # find the ID in the files 
#            ind = file_ADNI_RIDs.index(ADNI_RID)
#    
#            img = nib.load(data_dir + files[ind])
#            img_data = img.get_data()
#            img_data[img_data > 1.0] = 1.0
#            img_data[img_data < 0.0] = 0.0
#            histogram_data[k, :] = np.array(SPMK_3d.build_histogram_at_multiple_levels_rec_3d(SPMK_3d.quantise_intensities(img_data, n_intensity_levels), n_histogram_levels, 0, n_intensity_levels, [])).flatten()
#        
#        K = SPMK_3d.histogram_intersection_kernel(histogram_data, False)
#        np.savetxt(kernel_dir + 'SPMK_intersection_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '_test_90_1.csv', K, delimiter=',')
#        K = SPMK_3d.bhattacharyya_coefficient_kernel(histogram_data, False)
#        np.savetxt(kernel_dir + 'SPMK_bhattacharyya_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '_test_90_1.csv', K, delimiter=',')
#        
#        ## check this!!
#        K = np.dot(histogram_data, np.transpose(histogram_data))
#        np.savetxt(kernel_dir + 'SPMK_lin_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '_test_90_1.csv', K, delimiter=',')
        
        ################################################################################ 
        #### IF WE BUILD KERNEL IN MINIBATCHES - COMPROMISE IN TIME/MEMORY TRADEOFF ####
        ################################################################################    
                                                   
        step_size = 40
        
        
        K_histogram_intersection = np.zeros((len(ADNI_metadata), len(ADNI_metadata)))
        K_bhattacharyya = np.zeros((len(ADNI_metadata), len(ADNI_metadata)))
        K_SPMK_lin = np.zeros((len(ADNI_metadata), len(ADNI_metadata))) 
        
        for k in range((len(ADNI_metadata)/step_size) + 1) :
            
            start_ind_1 = k * step_size
            stop_ind_1 = min((start_ind_1 + step_size, len(ADNI_metadata)))
                                    
            print 'k = ' + str(k)
            print start_ind_1, stop_ind_1   
            
            histogram_data_1 = np.zeros(((int(stop_ind_1 - start_ind_1)), histogram_width))
            
            for m in range(start_ind_1, stop_ind_1) :

                if m % 10 == 0 :   
    
                    print m
    
                # find the ID in the files
                ADNI_RID = ADNI_RIDs[m]
                ind = file_ADNI_RIDs.index(ADNI_RID)
    
                img = nib.load(data_dir + files[ind])
                img_data = img.get_data()
                img_data[img_data > 1.0] = 1.0
                img_data[img_data < 0.0] = 0.0
                histogram_data_1[m - start_ind_1, :] = np.array(SPMK_3d.build_histogram_at_multiple_levels_rec_3d(SPMK_3d.quantise_intensities(img_data, n_intensity_levels), n_histogram_levels, 0, n_intensity_levels, [])).flatten()
            
            for l in range((len(ADNI_metadata)/step_size) + 1) :
                
                start_ind_2 = l * step_size
                stop_ind_2 = min((start_ind_2 + step_size, len(ADNI_metadata)))
                
                print 'l = ' + str(l)
                print start_ind_2, stop_ind_2
                
                if k == l :
                    
                    K_SPMK_lin[start_ind_1:stop_ind_1, start_ind_1:stop_ind_1] = np.dot(histogram_data_1, np.transpose(histogram_data_1))
                    K_histogram_intersection[start_ind_1:stop_ind_1, start_ind_1:stop_ind_1] = SPMK_3d.histogram_intersection_kernel_2(histogram_data_1, histogram_data_1, False)
                    K_bhattacharyya[start_ind_1:stop_ind_1, start_ind_1:stop_ind_1] = SPMK_3d.bhattacharyya_coefficient_kernel_2(histogram_data_1, histogram_data_1, False)
                    
                else :
                    
                    histogram_data_2 = np.zeros((int(stop_ind_2 - start_ind_2), histogram_width))
                    for m in range(start_ind_2, stop_ind_2) :

                        if m % 10 == 0 :   
    
                            print m
    
                        # find the ID in the files
                        ADNI_RID = ADNI_RIDs[m]
                        ind = file_ADNI_RIDs.index(ADNI_RID)
    
                        img = nib.load(data_dir + files[ind])
                        img_data = img.get_data()
                        img_data[img_data > 1.0] = 1.0
                        img_data[img_data < 0.0] = 0.0
                        histogram_data_2[m - start_ind_2, :] = np.array(SPMK_3d.build_histogram_at_multiple_levels_rec_3d(SPMK_3d.quantise_intensities(img_data, n_intensity_levels), n_histogram_levels, 0, n_intensity_levels, [])).flatten()
                    
                    K_SPMK_lin[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = np.dot(histogram_data_1, np.transpose(histogram_data_2))
                    K_histogram_intersection[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = SPMK_3d.histogram_intersection_kernel_2(histogram_data_1, histogram_data_2, False)
                    K_bhattacharyya[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = SPMK_3d.bhattacharyya_coefficient_kernel_2(histogram_data_1, histogram_data_2, False)
    
        np.savetxt(kernel_dir + 'SPMK_intersection_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', K_histogram_intersection, delimiter=',')
        np.savetxt(kernel_dir + 'SPMK_bhattacharyya_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', K_bhattacharyya, delimiter=',')
        np.savetxt(kernel_dir + 'SPMK_lin_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', K_SPMK_lin, delimiter=',')