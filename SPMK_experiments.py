# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:37:51 2016

@author: jonyoung
"""

# import the stuff we need
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.spatial.distance import pdist, squareform
import SPMK_3d

# parameters
n_intensity_levels = 16
n_histogram_levels = 6
# directories
data_dir = '/home/jonyoung/IoP_data/Data/IXI/GM-masked-groupwise-sub1pt5/'

# try loading an image
example_nii = nib.load(data_dir + 'IXI012-GM-masked-groupwise-sub1pt5.nii.gz')
example_nii_data = example_nii.get_data()
example_nii_slice = example_nii_data[:,:,40]

files = os.listdir(data_dir)
histograms = np.zeros((100, 599184))
for i, file in zip(range(100), files[:100]):
    
    img = nib.load(data_dir + file)
    img_data = img.get_data()
    img_data[img_data > 1.0] = 1.0
    img_codes = quantise_intensities(img_data, n_intensity_levels)
    histograms[i,:] =  np.array(build_histogram_at_multiple_levels_rec_3d(img_codes, n_histogram_levels, 0, n_intensity_levels, [])).flatten()

start = time.clock()
foo = histogram_intersection_kernel_1(histograms, True)
stop = time.clock()
print stop - start
print foo
print np.shape(foo)
start = time.clock()
foo = histogram_intersection_kernel_2(histograms, True)
stop = time.clock()
print stop - start
print foo
print np.shape(foo)
print np.min(np.min(foo))
print np.histogram(foo)