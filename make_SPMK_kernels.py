# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:04:27 2016

@author: jonyoung
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:37:51 2016

@author: jonyoung
"""

# import the stuff we need
import nibabel as nib
import pandas as pd
import os
#import SPMK_3d
import numpy as np
from sklearn import svm, cross_validation
import SPMK_3d

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

# create an array to hold all the image data and read the images into it
all_image_data = np.zeros((len(IXI_metadata), 2890000))
for i, row in zip(range(len(IXI_metadata)), IXI_metadata.iterrows()) :
    1
    if i % 10 == 0 :   
    
        print i
    
    IXI_ID = row[1][0]

    # find the ID in the files 
    ind = file_IXI_ids.index(IXI_ID)
    
    img = nib.load(data_dir + files[ind])
    #print i
    img_data = img.get_data()
    img_data[img_data > 1.0] = 1.0
    all_image_data[i,:] = img_data.flatten()
    
#all_image_data = all_image_data
    
#K = np.dot(all_image_data, np.transpose(all_image_data))
#np.savetxt(metadata_dir + 'kernels/K_lin.csv', K, delimiter=',')

for i in range(5, 6) :
    
    for j in range(7, 8) :
        
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
        
#        histogram_data = np.array(map(lambda x: np.array(SPMK_3d.build_histogram_at_multiple_levels_rec_3d(SPMK_3d.quantise_intensities(np.reshape(x, (170, 170, 100)), n_intensity_levels), n_histogram_levels, 0, n_intensity_levels, [])).flatten(), all_image_data))
#        K = SPMK_3d.histogram_intersection_kernel(histogram_data, False)
#        np.savetxt(metadata_dir + 'kernels/SPMK_intersection_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', K, delimiter=',')
#        K = SPMK_3d.bhattacharyya_coefficient_kernel(histogram_data, False)
#        np.savetxt(metadata_dir + 'kernels/SPMK_bhattacharyya_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', K, delimiter=',')
#        
#        ## check this!!
#        K = np.dot(histogram_data, np.transpose(histogram_data))
#        np.savetxt(metadata_dir + 'kernels/SPMK_lin_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '.csv', K, delimiter=',')
        
        ########################################################################### 
        #### IF WE BUILD KERNEL ENTRY BY ENTRY - SLOW BUT NOT MEMORY INTENSIVE ####
        ###########################################################################
        
#        K_histogram_intersection = np.zeros((len(all_image_data), len(all_image_data)))
#        K_bhattacharyya = np.zeros((len(all_image_data), len(all_image_data)))
#        K_SPMK_lin = np.zeros((len(all_image_data), len(all_image_data)))         
#        
#        for k in range(len(all_image_data)) :
#            
#            for l in range(len(all_image_data)) :
#                                                                    
#                img_data_1 = np.reshape(all_image_data[k,:], (170, 170, 100))                
#                img_data_2 = np.reshape(all_image_data[l,:], (170, 170, 100))  
#                img_codes_1 = SPMK_3d.quantise_intensities(img_data_1, n_intensity_levels)
#                img_codes_2 = SPMK_3d.quantise_intensities(img_data_2, n_intensity_levels)
#                histogram_data_1 = np.array(SPMK_3d.build_histogram_at_multiple_levels_rec_3d(img_codes_1, n_histogram_levels, 0, n_intensity_levels, [])).flatten()
#                histogram_data_2 = np.array(SPMK_3d.build_histogram_at_multiple_levels_rec_3d(img_codes_2, n_histogram_levels, 0, n_intensity_levels, [])).flatten()                
#                K_SPMK_lin[k, l] = np.dot(histogram_data_1, histogram_data_2)               
#                K_histogram_intersection[k, l] = np.minimum(histogram_data_1,histogram_data_2).sum()
#                K_bhattacharyya[k, l] = np.sqrt(np.multiply(histogram_data_1,histogram_data_2)).sum()
#                
#                ### extra checks
#                if (k == 0 or k == 1) and (l == 0 or l == 1) :
#                    
#                    print 'k = ' + str(k) + ', l = ' + str(l)    
#                    print histogram_data_1
#                    print histogram_data_2
#                    foo = np.multiply(histogram_data_1, histogram_data_2).sum()
#                    print foo
#                
#                
#        # save kernels
#        np.savetxt(metadata_dir + 'kernels/SPMK_intersection_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '_2.csv', K_histogram_intersection, delimiter=',')
#        np.savetxt(metadata_dir + 'kernels/SPMK_bhattacharyya_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '_2.csv', K_bhattacharyya, delimiter=',')
#        np.savetxt(metadata_dir + 'kernels/SPMK_lin_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '_2.csv', K_SPMK_lin, delimiter=',')


        ################################################################################ 
        #### IF WE BUILD KERNEL IN MINIBATCHES - COMPROMISE IN TIME/MEMORY TRADEOFF ####
        ################################################################################    
                                                   
        step_size = 40
        
        
        K_histogram_intersection = np.zeros((len(all_image_data), len(all_image_data)))
        K_bhattacharyya = np.zeros((len(all_image_data), len(all_image_data)))
        K_SPMK_lin = np.zeros((len(all_image_data), len(all_image_data))) 
        
        for k in range((len(all_image_data)/step_size) + 1) :
            
            start_ind_1 = k * step_size
            stop_ind_1 = min((start_ind_1 + step_size, len(all_image_data)))
                       
            
            print 'k = ' + str(k)
            print start_ind_1, stop_ind_1
            
            histogram_data_1 = np.zeros((int(stop_ind_1 - start_ind_1), histogram_width))         
            histogram_data_1 = map(lambda x: np.array(SPMK_3d.build_histogram_at_multiple_levels_rec_3d(SPMK_3d.quantise_intensities(np.reshape(x, (170, 170, 100)), n_intensity_levels), n_histogram_levels, 0, n_intensity_levels, [])).flatten(), all_image_data[start_ind_1:stop_ind_1, :])                                    
            
            for l in range((len(all_image_data)/step_size) + 1) :
                
                start_ind_2 = l * step_size
                stop_ind_2 = min((start_ind_2 + step_size, len(all_image_data)))
                
                print 'l = ' + str(l)
                print start_ind_2, stop_ind_2
                
                if k == l :
                    
                    K_SPMK_lin[start_ind_1:stop_ind_1, start_ind_1:stop_ind_1] = np.dot(histogram_data_1, np.transpose(histogram_data_1))
                    K_histogram_intersection[start_ind_1:stop_ind_1, start_ind_1:stop_ind_1] = SPMK_3d.histogram_intersection_kernel_2(histogram_data_1, histogram_data_1, False)
                    K_bhattacharyya[start_ind_1:stop_ind_1, start_ind_1:stop_ind_1] = SPMK_3d.bhattacharyya_coefficient_kernel_2(histogram_data_1, histogram_data_1, False)
                                     
                    
                    
                else :
                    
                    histogram_data_2 = np.zeros((int(stop_ind_2 - start_ind_2), histogram_width))         
                    histogram_data_2 = map(lambda x: np.array(SPMK_3d.build_histogram_at_multiple_levels_rec_3d(SPMK_3d.quantise_intensities(np.reshape(x, (170, 170, 100)), n_intensity_levels), n_histogram_levels, 0, n_intensity_levels, [])).flatten(), all_image_data[start_ind_2:stop_ind_2, :])                                    
                    K_SPMK_lin[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = np.dot(histogram_data_1, np.transpose(histogram_data_2))
                    K_histogram_intersection[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = SPMK_3d.histogram_intersection_kernel_2(histogram_data_1, histogram_data_2, False)
                    K_bhattacharyya[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = SPMK_3d.bhattacharyya_coefficient_kernel_2(histogram_data_1, histogram_data_2, False)
        
        # save kernels
        np.savetxt(metadata_dir + 'kernels/SPMK_intersection_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '_3.csv', K_histogram_intersection, delimiter=',')
        np.savetxt(metadata_dir + 'kernels/SPMK_bhattacharyya_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '_3.csv', K_bhattacharyya, delimiter=',')
        np.savetxt(metadata_dir + 'kernels/SPMK_lin_' + str(n_intensity_levels) + '_' + str(n_histogram_levels) + '_3.csv', K_SPMK_lin, delimiter=',')                
                

                
        
        
        
  

