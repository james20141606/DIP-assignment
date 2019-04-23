import numpy as np
import cv2
import scipy.sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm_notebook as tqdm
from os import path

def laplacian_matrix(n, m):
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    return mat_A


def poisson_edit(height_insert, width_insert,alpha,permc_spec,padding_mask_xmin, padding_mask_xmax,padding_mask_ymin,padding_mask_ymax, source, target, mask):
    # Assume: 
    # target is not smaller than source.
    # shape of mask is same as shape of target.
    # permc_spec NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD
    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0

    y_min_, y_max_, x_min_, x_max_ = np.min(np.where(mask!=0)[0]),np.max(np.where(mask!=0)[0]),np.min(np.where(mask!=0)[1]),np.max(np.where(mask!=0)[1])

    x_range = x_max - x_min
    y_range = y_max - y_min
    x_range_ = x_max_ - x_min_
    y_range_ = y_max_ - y_min_
        
    source_ = np.zeros([target.shape[0],target.shape[1],target.shape[2]])
    mask_ = np.zeros([target.shape[0],target.shape[1]])
    source_[height_insert-padding_mask_ymin:height_insert+y_range_+padding_mask_ymax,width_insert-padding_mask_xmin:width_insert+x_range_+padding_mask_xmax,:]=source[y_min_-padding_mask_ymin:y_max_+padding_mask_ymax, x_min_-padding_mask_xmin:x_max_+padding_mask_xmax,:]
    mask_[height_insert-padding_mask_ymin:height_insert+y_range_+padding_mask_ymax,width_insert-padding_mask_xmin:width_insert+x_range_+padding_mask_xmax]=mask[y_min_-padding_mask_ymin:y_max_+padding_mask_ymax, x_min_-padding_mask_xmin:x_max_+padding_mask_xmax]

    source=source_
    mask=mask_

    mask = mask[y_min:y_max, x_min:x_max]    
    mask[mask != 0] = 1
    #mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    
    mat_A = laplacian_matrix(y_range, x_range)

    # for \Delta g
    laplacian = mat_A.tocsc()

    # set the region outside the mask to identity    
    for y in tqdm(range(1, y_range - 1)):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0

    mat_A = mat_A.tocsc()

    mask_flat = mask.flatten()  
    target_ = target.copy()  
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()        
   
 
        mat_b = laplacian.dot(source_flat)*alpha

        mat_b[mask_flat==0] = target_flat[mask_flat==0]
        
        x = spsolve(mat_A, mat_b,permc_spec=permc_spec)

        x = x.reshape((y_range, x_range))

        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')

        target_[y_min:y_max, x_min:x_max, channel] = x

    return target_

