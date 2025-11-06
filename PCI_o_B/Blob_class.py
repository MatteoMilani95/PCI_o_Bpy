# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 11:15:10 2025

@author: matte
"""

from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from glob import glob
import os
import matplotlib.pylab as pl
from skimage import img_as_ubyte
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter, median_filter
from skimage.feature import peak_local_max
from scipy.interpolate import interp1d
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from scipy.ndimage import label
from skimage.feature import blob_log
from skimage import io
from skimage.measure import find_contours, regionprops
from matplotlib.path import Path
from skimage.measure import EllipseModel
from skimage import io, color, filters, measure, morphology
from PCI_o_B import color_functions as cf


class BLOB():
    
    def __init__(self, blob_image=0, blob_mask=0):
        self.blob_image = blob_image
        self.blob_mask = blob_mask

                #self.Input_101 =[]
    def __repr__(self): 
        return '<ROI: fn%s>' % (self.path)
    
    def get_init_args(self):
        return (self.blob_image, self.blob_mask)
            
    def __str__(self):
        str_res  = '\n|---------------|'
        str_res += '\n| BLOB class: '
        str_res += '\n|--------------------+--------------------|'
        return str_res
    
    
    def detect_blob(self,image):
        image_smoothed = filters.gaussian(image, sigma=2)
        
        # Use Otsu's method to threshold the image
        thresh = filters.threshold_otsu(image_smoothed)
        binary = image_smoothed > thresh
        
        # Perform morphological closing to fill small holes (adjust the disk size as needed)
        binary_closed = morphology.closing(binary, morphology.disk(3))
        
        # Remove small objects that are not likely to be the large blob
        binary_cleaned = morphology.remove_small_objects(binary_closed, min_size=500)
        
        # Label connected regions
        labels = measure.label(binary_cleaned)
        regions = measure.regionprops(labels)
        
        # If multiple regions are detected, select the largest one (assuming that's the blob)
        if regions:
            self.largest_region = max(regions, key=lambda r: r.area)
        else:
            self.largest_region = None
            self.max_value = np.nan
            self.max_pos = (np.nan, np.nan)
            
        if self.largest_region is not None:
            # Extract the bounding box of the blob
            minr, minc, maxr, maxc = self.largest_region.bbox
            self.blob_image = image[minr:maxr, minc:maxc]
        
            # Create a mask for the blob within the bounding box
            self.blob_mask = (labels[minr:maxr, minc:maxc] == self.largest_region.label)
        
            # Compute the maximum intensity and its position within the blob region
            # Mask out values not belonging to the blob
            blob_values = self.blob_image.astype(float).copy()
            blob_values[~self.blob_mask] = -np.inf  # So they won't be chosen as maximum
            max_idx = np.unravel_index(np.argmax(blob_values), blob_values.shape)
            self.max_value = blob_values[max_idx]
            # Convert position to full image coordinates
            self.max_pos = (max_idx[0] + minr, max_idx[1] + minc)
        
            # Define an iso-level (for example, 50% of the peak intensity)
            iso_level = 0.55 * self.max_value
        
            # Find iso-height contours within the blob region at the defined iso-level.
            # Note: find_contours works on the full image array so we apply it on the cropped blob.
            contours = find_contours(self.blob_image, level=iso_level)
        
            if contours:
                # Choose the longest contour (if multiple are detected)
                iso_contour = max(contours, key=lambda arr: arr.shape[0])
                # Shift contour coordinates to full image coordinates
                iso_contour_full = iso_contour + np.array([minr, minc])
        
                # Calculate distances from the peak (max_pos) to each point on the iso contour
                distances = np.sqrt((iso_contour_full[:, 0] - self.max_pos[0])**2 + (iso_contour_full[:, 1] - self.max_pos[1])**2)
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)
                self.symmetry_metric = std_distance / mean_distance  # lower is more symmetric
       
        return
        

    


    
    def get_oblatio_blob(self,iso_percents_min,iso_percents_max,iso_percents_num):
        
        
        iso_percents = np.linspace(iso_percents_min,iso_percents_max,iso_percents_num)
        closed_thresh = 100  # maximum allowed gap between first and last contour point
        
        
        # Convert the blob image to float and mask out non-blob pixels
        blob_values = self.blob_image.astype(float)
        blob_values[~self.blob_mask] = -np.inf
        
        # Get the peak value from the blob (for iso-level calculations)
        max_value = np.max(blob_values)
        
        # For each iso percent, extract contour, check closure, fit ellipse, and plot results

        xc_s = []
        yc_s = []
        a_s   = []
        b_s   = []
        theta_s = []
        oblate_values = []
        for i, perc in enumerate(iso_percents):
            iso_level = perc * max_value
            # Find contours at this iso level
            contours = find_contours(self.blob_image, level=iso_level)
            if not contours:
                continue
            # Choose the longest contour as the candidate iso-perimeter
            iso_contour = max(contours, key=lambda arr: arr.shape[0])
            # Check if the contour is closed (first and last points close enough)
            if np.linalg.norm(iso_contour[0] - iso_contour[-1]) > closed_thresh:
                continue
        
            # Shift contour coordinates to full-image space (using the blob bounding box)
            minr, minc, _, _ = self.largest_region.bbox
            iso_contour_full = iso_contour + np.array([minr, minc])
            
            # Fit an ellipse to the iso-contour and compute oblateness
            ellipse_params, oblate_value = cf.fit_ellipse_and_oblateness(iso_contour_full, normalization='difference')
            if ellipse_params is None:
                continue
            xc, yc, a, b, theta = ellipse_params

            xc_s.append(xc)
            yc_s.append(yc)
            a_s.append(a)
            b_s.append(b)
            theta_s.append(theta)
            oblate_values.append(oblate_value)

        return xc_s, yc_s, a_s, b_s, theta_s, oblate_values,iso_contour_full
    
    def get_oblatio_blob_2(self,iso_level):
        closed_thresh = 30
# Convert the blob image to float and mask out non-blob pixels
        blob_values = self.blob_image.astype(float)
        blob_values[~self.blob_mask] = -np.inf
        
        # Get the peak value from the blob (for iso-level calculations)
        
        # Find contours at this iso level
        contours = find_contours(self.blob_image, level=iso_level)
        if not contours:
            print('fagiano 1')
            xc = np.nan
            yc = np.nan
            a = np.nan
            b = np.nan
            theta = np.nan
            oblate_value = np.nan
            iso_contour_full = np.nan
            return xc, yc, a, b, theta, oblate_value,iso_contour_full

            # Choose the longest contour as the candidate iso-perimeter
        iso_contour = max(contours, key=lambda arr: arr.shape[0])
        


        
# Shift contour coordinates to full-image space (using the blob bounding box)
        minr, minc, _, _ = self.largest_region.bbox
        iso_contour_full = iso_contour + np.array([minr, minc])
            
        # Fit an ellipse to the iso-contour and compute oblateness
        ellipse_params, oblate_value = cf.fit_ellipse_and_oblateness(iso_contour_full, normalization='difference')
        if ellipse_params is None:
            xc = np.nan
            yc = np.nan
            a = np.nan
            b = np.nan
            theta = np.nan
            oblate_value = np.nan
            return
        
        xc, yc, a, b, theta = ellipse_params

        return xc, yc, a, b, theta, oblate_value,iso_contour_full