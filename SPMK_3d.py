# import the stuff we need
import numpy as np
from scipy.spatial.distance import pdist, squareform

def build_histogram(codes, n_levels):
    codes_flat = codes.flat
    histogram = np.zeros(n_levels)
    for code in codes_flat:
        histogram[code] += 1
    return histogram    

# function to bin image intensity values into n_levels bins
def quantise_intensities(image_data, n_levels):
    
    bins = 1.0/n_levels * np.array(range(1, n_levels + 1))
    return np.digitize(image_data, bins, right=True)
    
def build_histogram_at_multiple_levels_rec_2d(codes_image, n_histogram_levels, level, n_intensity_levels, final_histogram):
            
    # lowest level
    lowest_level = n_histogram_levels - 1

    # if level = n_levels-1 we have reached the bottom level of the pyramid - base case
    # divide up image into 4 and calculate a histogram for each
    if lowest_level == level :
                            
        # dimensions of code (sub)image    
        width = codes_image.shape[0]
        height = codes_image.shape[1]
    
        # (lowest) step sizes
        width_step = int(width/2)
        height_step = int(height/2)
                              
        # caculate histograms for the four subimages
        raw_histogram_0 = build_histogram(codes_image[0:width_step, 0:height_step], n_intensity_levels)
        raw_histogram_1 = build_histogram(codes_image[0:width_step, height_step:height], n_intensity_levels)
        raw_histogram_2 = build_histogram(codes_image[width_step:width, 0:height_step], n_intensity_levels)
            aw_histogram_3 = build_histogram(codes_image[width_step:width, height_step:height], n_intensity_levels)
            
        # calculate weight for this level
        if level == 0 :
            
            weight = 1.0 / (2 ** (lowest_level))
            
        else :
            
            weight = 1.0 / (2 ** (lowest_level - level + 1))
                
        print level, weight
            
        # then calculate raw histogram as sum of subimage histograms
        raw_histogram = raw_histogram_0 + raw_histogram_1 + raw_histogram_2 + raw_histogram_3
            
        # calculate weighted histogram and append it to final histogram
        weighted_histogram = raw_histogram * weight
        final_histogram.append(weighted_histogram)
            
        print 'raw histogram: '
        print raw_histogram
        print 'weighted histogram: '
        print weighted_histogram
                     
        # to avoid needing a dummy variable, if level is zero, retu1168rn final only histogram to calling level above
        if level == 0 :
            return final_histogram
            # otherwise return histogram for this level and  final histogram to calling level above
        else :
            return raw_histogram,final_histogram
        
    # if not, level < n_levels. Calculate histogram at this level as a sum of
    # recursive calls to the next level below
    else :

        
        # dimensions of code (sub)image    
        width = codes_image.shape[0]
        height = codes_image.shape[1]
    
        # (lowest) step sizes
        width_step = int(width/2)
        height_step = int(height/2)
        
        # get histograms for the four subimages with recursive calls
        raw_histogram_0,final_histogram = build_histogram_at_multiple_levels_rec_2d(codes_image[0:width_step, 0:height_step], n_histogram_levels, level + 1, n_intensity_levels, final_histogram)
        raw_histogram_1,final_histogram = build_histogram_at_multiple_levels_rec_2d(codes_image[0:width_step, height_step:height], n_histogram_levels, level + 1, n_intensity_levels, final_histogram)
        raw_histogram_2,final_histogram = build_histogram_at_multiple_levels_rec_2d(codes_image[width_step:width, 0:height_step], n_histogram_levels, level + 1, n_intensity_levels, final_histogram)
        raw_histogram_3,final_histogram = build_histogram_at_multiple_levels_rec_2d(codes_image[width_step:width, height_step:height], n_histogram_levels, level + 1, n_intensity_levels, final_histogram)
                                  
        # calculate weight for this level
        if level == 0 :
            
            weight = 1.0 / (2 ** (lowest_level))
            
        else :
            
            weight = 1.0 / (2 ** (lowest_level - level + 1))
                
        print level, weight
            
        # then calculate raw histogram as sum of subimage histograms
        raw_histogram = raw_histogram_0 + raw_histogram_1 + raw_histogram_2 + raw_histogram_3
            
        # calculate weighted histogram and append it to final histogra1168m
        weighted_histogram = raw_histogram * weight
        final_histogram.append(weighted_histogram)
            
        print 'raw histogram: '
        print raw_histogram
        print 'weighted histogram: '
        print weighted_histogram
        
        # to avoid needing a dummy variable, if level is zero, return only the final histogram to calling level above
        if level == 0 :
            return final_histogram            
        # otherwise return histogram and final histogram to calling level above
        else:
            return raw_histogram, final_histogram
                
def build_histogram_at_multiple_levels_rec_3d(codes_image, n_histogram_levels, level, n_intensity_levels, final_histogram):
            
    # lowest level
    lowest_level = n_histogram_levels - 1

    # if level = n_levels-1 we have reached the bottom level of the pyramid - base case
    # divide up image into 8 and calculate a histogram for each
    if lowest_level == level :
                            
        # dimensions of code (sub)image    
        width = codes_image.shape[0]
        height = codes_image.shape[1]
        depth = codes_image.shape[2]
    
        # (lowest) step sizes
        width_step = int(width/2)
        height_step = int(height/2)
        depth_step = int(depth/2)
                              
        # caculate histograms for the eight subimages
        raw_histogram_0 = build_histogram(codes_image[0:width_step, 0:height_step, 0:depth_step], n_intensity_levels)
        raw_histogram_1 = build_histogram(codes_image[0:width_step, height_step:height, 0:depth_step], n_intensity_levels)
        raw_histogram_2 = build_histogram(codes_image[width_step:width, 0:height_step, 0:depth_step], n_intensity_levels)
        raw_histogram_3 = build_histogram(codes_image[width_step:width, height_step:height, 0:depth_step], n_intensity_levels)
        raw_histogram_4 = build_histogram(codes_image[0:width_step, 0:height_step, depth_step:depth], n_intensity_levels)
        raw_histogram_5 = build_histogram(codes_image[0:width_step, height_step:height, depth_step:depth], n_intensity_levels)
        raw_histogram_6 = build_histogram(codes_image[width_step:width, 0:height_step, depth_step:depth], n_intensity_levels)
        raw_histogram_7 = build_histogram(codes_image[width_step:width, height_step:height, depth_step:depth], n_intensity_levels)
            
        # calculate weight for this level
        if level == 0 :
            
            weight = 1.0 / (2 ** (lowest_level))
            
        else :
            
            weight = 1.0 / (2 ** (lowest_level - level + 1))
                
        #print level, weight
            
        # then calculate raw histogram as sum of subimage histograms
        raw_histogram = raw_histogram_0 + raw_histogram_1 + raw_histogram_2 + raw_histogram_3 + raw_histogram_4 + raw_histogram_5 + raw_histogram_6 + raw_histogram_7
            
        # calculate weighted histogram and append it to final histogram
        weighted_histogram = raw_histogram * weight
        final_histogram.append(weighted_histogram)
            
        #print 'raw histogram: '
        #print raw_histogram
        #print 'weighted histogram: '
        #print weighted_histogram
                     
        # to avoid needing a dummy variable, if level is zero, return final only histogram to calling level above
        if level == 0 :
            return final_histogram
        # otherwise return histogram for this level and  final histogram to calling level above
        else :
            return raw_histogram,final_histogram
        
    # if not, level < n_levels. Calculate histogram at this level as a sum of
    # recursive calls to the next level below
    else :

        # dimensions of code (sub)image    
        width = codes_image.shape[0]
        height = codes_image.shape[1]
        depth = codes_image.shape[2]
    
        # (lowest) step sizes
        width_step = int(width/2)
        height_step = int(height/2)
        depth_step = int(depth/2)
        
        # get histograms for the four subimages with recursive calls
        raw_histogram_0,final_histogram = build_histogram_at_multiple_levels_rec_3d(codes_image[0:width_step, 0:height_step, 0:depth_step], n_histogram_levels, level + 1, n_intensity_levels, final_histogram)
        raw_histogram_1,final_histogram = build_histogram_at_multiple_levels_rec_3d(codes_image[0:width_step, height_step:height, 0:depth_step], n_histogram_levels, level + 1, n_intensity_levels, final_histogram)
        raw_histogram_2,final_histogram = build_histogram_at_multiple_levels_rec_3d(codes_image[width_step:width, 0:height_step, 0:depth_step], n_histogram_levels, level + 1, n_intensity_levels, final_histogram)
        raw_histogram_3,final_histogram = build_histogram_at_multiple_levels_rec_3d(codes_image[width_step:width, height_step:height, 0:depth_step], n_histogram_levels, level + 1, n_intensity_levels, final_histogram)
        raw_histogram_4,final_histogram = build_histogram_at_multiple_levels_rec_3d(codes_image[0:width_step, 0:height_step, depth_step:depth], n_histogram_levels, level + 1, n_intensity_levels, final_histogram)
        raw_histogram_5,final_histogram = build_histogram_at_multiple_levels_rec_3d(codes_image[0:width_step, height_step:height, depth_step:depth], n_histogram_levels, level + 1, n_intensity_levels, final_histogram)
        raw_histogram_6,final_histogram = build_histogram_at_multiple_levels_rec_3d(codes_image[width_step:width, 0:height_step, depth_step:depth], n_histogram_levels, level + 1, n_intensity_levels, final_histogram)
        raw_histogram_7,final_histogram = build_histogram_at_multiple_levels_rec_3d(codes_image[width_step:width, height_step:height, depth_step:depth], n_histogram_levels, level + 1, n_intensity_levels, final_histogram)                      
            
        # calculate weight for this level
        if level == 0 :
            
            weight = 1.0 / (2 ** (lowest_level))
            
        else :
            
            weight = 1.0 / (2 ** (lowest_level - level + 1))
                
        #print level, weight
            
        # then calculate raw histogram as sum of subimage histograms
        raw_histogram = raw_histogram_0 + raw_histogram_1 + raw_histogram_2 + raw_histogram_3 + raw_histogram_4 + raw_histogram_5 + raw_histogram_6 + raw_histogram_7
            
        # calculate weighted histogram and append it to final histogram
        weighted_histogram = raw_histogram * weight
        final_histogram.append(weighted_histogram)
            
        #print 'raw histogram: '
        #print raw_histogram
        #print 'weighted histogram: '
        #print weighted_histogram
        
        # to avoid needing a dummy variable, if level is zero, return final only histogram to calling level above
        if level == 0 :
            return final_histogram            
        # otherwise return histogram and final histogram to calling level above
        else:
            return raw_histogram, final_histogram
                            
def histogram_intersection_kernel_1(histograms, normalise=True):
    
    if normalise :
        
        histogram_total = sum(histograms[0,:])
        histograms = histograms / histogram_total
        
    n = np.shape(histograms)[0]

    
    K = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            K[i,j] = np.minimum(histograms[i,:], histograms[j,:]).sum()
            
    return K
    
def histogram_intersection_kernel_2(histograms, normalise=True):
    
    if normalise :
        
        histogram_total = sum(histograms[0,:])
        histograms = histograms / histogram_total
        
    K = squareform(pdist(histograms, lambda x, y: np.minimum(x,y).sum()))
    np.fill_diagonal(K, 1.0)
    return K
    
