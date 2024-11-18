# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:59:06 2024

@author: Matteo
"""

import numpy as np
import os
from os import walk
import matplotlib.pyplot as plt
import time
from PIL import Image
from scipy.ndimage import correlate
from PCI_o_B import SharedFunctions as sf
from scipy import signal
from scipy.signal import savgol_filter

class MULTPLE_CORRECTION():
    
    def __init__(self,normalization):
        """Initialize ROIs
        
        Parameters
        ----------
        Filename: complete filename of the CI file
        
        
        """
        
        self.intesity_norm = normalization

        
    def __repr__(self): 
        return '<ROI: fn%s>' % (self.path)
    
    def __str__(self):
        str_res  = '\n|---------------|'
        str_res += '\n| SAMLL ANGLE class: '
        str_res += '\n|--------------------+--------------------|'
        return str_res
    
    def load_image(self,inputfolder,img1_name,dark_folder):
        
            self.dark_folder = dark_folder
            if dark_folder == '':
                self.use_dark = False   #set to True only if dark images are available!
            else:
                self.use_dark = True
                
            img1_name = inputfolder + img1_name
            img2_name = img1_name
            
            if img1_name == img2_name:  #same image: will be autocorrelation
                self.autoc = True
            else: self.autoc = False  #will be cross correlation
            
        
            print('loading images...')
            # img1 = cv2.imread(img1_name, 0)
            # img2 = cv2.imread(img2_name, 0)
            self.img1 = Image.open(img1_name)
            self.image_float = np.asarray(self.img1,dtype=float)
            
    def calc_correlation(self, xtopleft,ytopleft,xsize,ysize,shift_x,shift_y,axis=0):
        
            fig, ax = plt.subplots()
            x = range(300)
            ax.imshow(self.img1, extent=[0, 2048, 1088, 0])
            
            plt.vlines(xtopleft, ymin=ytopleft, ymax=ytopleft+ysize,color='red')
            plt.vlines(xtopleft+xsize, ymin=ytopleft, ymax=ytopleft+ysize,color='red')
            plt.hlines(ytopleft, xmin=xtopleft,xmax=xtopleft+xsize,color='red')
            plt.hlines(ytopleft+ysize, xmin=xtopleft,xmax=xtopleft+xsize,color='red')
            
            
            plt.vlines(xtopleft+shift_x, ymin=ytopleft+shift_y, ymax=ytopleft+ysize+shift_y,color='orange')
            plt.vlines(xtopleft+xsize+shift_x, ymin=ytopleft+shift_y, ymax=ytopleft+ysize+shift_y,color='orange')
            plt.hlines(ytopleft+shift_y, xmin=xtopleft+shift_x,xmax=xtopleft+xsize+shift_x,color='orange')
            plt.hlines(ytopleft+ysize+shift_y, xmin=xtopleft+shift_x,xmax=xtopleft+xsize+shift_x,color='orange')
            
         
            img1_arr = np.asarray(self.img1,dtype=float)
            
            
            
            width, height = self.img1.size

            # Create a new blank image with the same size and the same mode
            shifted_image = Image.new(self.img1.mode, (width, height))
            
            # Paste the original image into the new image, shifted down by 14 pixels
            shifted_image.paste(self.img1, (shift_x, shift_y))
            
            
            #img2_arr = np.asarray(self.img2,dtype=float)

            img2_arr = np.asarray(shifted_image,dtype=float)
            

            
            #cut to ROI:
            img1_arr = img1_arr[ytopleft:(ytopleft+ysize),xtopleft:(xtopleft+xsize)]
            img2_arr = img2_arr[ytopleft:(ytopleft+ysize),xtopleft:(xtopleft+xsize)]
            
            row_num = len(img1_arr)
            col_num = len(img2_arr[0])
            
            if (self.use_dark):
                self.dark_folder += '\\'
                dark_filenames = sf.FindFileNames(self.dark_folder)
                print(dark_filenames)
                dark_arr3D = np.zeros((len(dark_filenames), row_num, col_num))
                for i in range(len(dark_filenames)):
                    cur_dark = Image.open(self.dark_folder + dark_filenames[i])
                    dark_arr3D[i] = np.asarray(cur_dark,dtype = float)[ytopleft:(ytopleft+ysize),xtopleft:(xtopleft+xsize)]
                dark_avg = np.average(dark_arr3D, axis=0)
                self.img1_nobkg = img1_arr - dark_avg
                self.img2_nobkg = img2_arr - dark_avg
                
            else:
                self.img1_nobkg = img1_arr
                self.img2_nobkg = img2_arr
                
    
        #calculate (auto/cross)correlation   
            self.res_corr = signal.correlate(self.img1_nobkg, self.img2_nobkg)
                    
            #normalize with respect to number of pairs of values that contribute to each correlation value:
            norm = np.ones((ysize,xsize)) 
            self.corr_norm = signal.correlate(norm, norm)
            self.res_corr /= self.corr_norm
                
            #normalize with respect to average intensity, the way a (time) 
            #autocorrelation function is usually normalized
            self.i1ave = self.img1_nobkg.mean()
            self.i2ave = self.img2_nobkg.mean()
            self.res_corr /= (self.i1ave*self.i2ave)
            self.res_corr -= 1.
          
                    
            tstamp = time.asctime( time.localtime(time.time()) )
            tstamp = tstamp.replace(':','-')
            tstamp.replace(' ','_')
        

    
        #find peak position (pixel resolved)
            if self.autoc==True: #if autocorrelation: the 'real' peak must be at zero lag
                indy = int(self.res_corr.shape[0]/2)
                indx = int(self.res_corr.shape[1]/2)
            else:  #cross correlation: we don't know a priori where the peak is....
                ind = np.unravel_index(np.argmax(self.res_corr, axis=None), self.res_corr.shape)
                indy = ind[0]
                indx = ind[1]
            print ('Maximum of correlation: (Delta_x,Delta_y): ' + str(indx) + ',  ' +  str(indy))
            print ('Height of correlation peak: ' + str(self.res_corr[ indy, indx ]))
            #self.value_peak_correlation = self.res_corr[ indy, indx ]
            
            if axis == 0:
                
                self.ax = 0
            
                self.cut_x = self.res_corr[indy]
                self.cut_y = self.res_corr[:,indx]
                self.peaks,_ = signal.find_peaks(self.cut_x,height=0.1)
                print(self.peaks)
                
                length = len(self.peaks)
    
                # Ensure the array length is odd
                if length % 2 == 0:
                    raise ValueError("The array length must be odd")
                
                # Calculate the center index
                center_index = length // 2
                
                center_index
                
                self.h_0 = np.asarray(self.cut_x[self.peaks])[center_index]
                self.h = np.asarray(self.cut_x[self.peaks])[center_index-1]
                
                
                self.value_peak_max = np.max(np.asarray(self.cut_x[self.peaks]))
                
            else:
                self.ax = 1
                self.cut_x = self.res_corr[indy]
                self.cut_y = self.res_corr[:,indx]
                self.peaks,_ = signal.find_peaks(self.cut_y,height=0.1)
                
                
                length = len(self.peaks)
    
                # Ensure the array length is odd
                if length % 2 == 0:
                    raise ValueError("The array length must be odd")
                
                # Calculate the center index
                center_index = length // 2
                
                center_index
                
                self.h_0 = np.asarray(self.cut_y[self.peaks])[center_index]
                self.h = np.asarray(self.cut_y[self.peaks])[center_index-1]
                
                
                self.value_peak_max = np.max(np.asarray(self.cut_y[self.peaks]))
            

    def set_h_single(self, index):
        
        if self.ax ==0:
        
        
            self.value_peak_correlation  = np.asarray(self.cut_x[self.peaks])[index]
            
            return np.asarray(self.cut_x[self.peaks])[index]
        
        else:
            
            self.value_peak_correlation  = np.asarray(self.cut_y[self.peaks])[index]
            
            return np.asarray(self.cut_y[self.peaks])[index]
        
    def define_rois(self, roi_size, overlap):
        
        self.rois = []
        self.roi_size = roi_size
        step_size = roi_size - overlap
        
        image_width, image_height = self.img1.size
        
        for topy in range(0, image_height - roi_size + 1, step_size):
            for topx in range(0, image_width - roi_size + 1, step_size):
                self.rois.append((topx, topy))
        
        # Handle the edge cases to ensure all areas are covered
        if (image_width - roi_size) % step_size != 0:
            for topy in range(0, image_height - roi_size + 1, step_size):
                self.rois.append((image_width - roi_size, topy))
        
        if (image_height - roi_size) % step_size != 0:
            for topx in range(0, image_width - roi_size + 1, step_size):
                self.rois.append((topx, image_height - roi_size))
        
        if (image_width - roi_size) % step_size != 0 and (image_height - roi_size) % step_size != 0:
            self.rois.append((image_width - roi_size, image_height - roi_size))
        
        return 
            
            
        
    def calculate_avg_int(self):
        
        self.avg_I1 = self.i1ave/self.intesity_norm
        self.avg_I2 = self.i2ave/self.intesity_norm
        
        self.average_of_square= np.mean(self.img1_nobkg*self.img2_nobkg)/self.intesity_norm**2
        self.sqaure_of_avg_int = self.i1ave*self.i2ave/self.intesity_norm**2
        
        
    def show_illumination(self, ax=0):
        
        image_array = np.array(self.image_float)

# Pad the image to make it square
        size = max(image_array.shape)
        padded_image = np.pad(image_array, 
                              ((0, size - image_array.shape[0]), (0, size - image_array.shape[1])),
                              mode='constant', constant_values=0)
        
        # Perform 2D FFT
        fft_image = np.fft.fft2(padded_image)
        fft_shifted = np.fft.fftshift(fft_image)  # Shift the zero frequency component to the center
        
        # Compute the magnitude spectrum
        self.magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
        self.mid_col = self.magnitude_spectrum[:, self.magnitude_spectrum.shape[1] // 2]
        
        intensity_profile = np.mean(self.magnitude_spectrum, axis=ax)
        
        plt.figure()
        plt.imshow(self.magnitude_spectrum)
        
        plt.figure()
        plt.plot(intensity_profile)

     
        
        
        
        
        return
    
    def calc_correlation_2_0(self,axis=0):
        

            
         
            img1_arr = np.asarray(self.img1,dtype=float)
            
            
            
            width, height = self.img1.size

            # Create a new blank image with the same size and the same mode
            shifted_image = Image.new(self.img1.mode, (width, height))
            
            # Paste the original image into the new image, shifted down by 14 pixels
            shifted_image.paste(self.img1, (0, 0))
            
            
            #img2_arr = np.asarray(self.img2,dtype=float)

            img2_arr = np.asarray(shifted_image,dtype=float)
            
            
            self.h_0 = []
            self.h = []
            self.rho = []
                    
            self.Delta_x = []
            self.Delta_y = []
            
            self.I_avg = []
            
            
            self.Center_INDEX = []
            self.CUTX = []
            self.CUTY = []
            self.up = []
            self.down = []
            
            self.baseline = []
            
            self.dark_folder += '\\'

            for i in range(len(self.rois)):
                
                img1_arr = np.asarray(self.img1,dtype=float)
            
            
            
                width, height = self.img1.size
    
                # Create a new blank image with the same size and the same mode
                shifted_image = Image.new(self.img1.mode, (width, height))
                
                # Paste the original image into the new image, shifted down by 14 pixels
                shifted_image.paste(self.img1, (0, 0))
                
                
                #img2_arr = np.asarray(self.img2,dtype=float)
    
                img2_arr = np.asarray(shifted_image,dtype=float)
                #cut to ROI:
                img1_arr = img1_arr[self.rois[i][1]:(self.rois[i][1]+self.roi_size),self.rois[i][0]:(self.rois[i][0]+self.roi_size)]
                img2_arr = img2_arr[self.rois[i][1]:(self.rois[i][1]+self.roi_size),self.rois[i][0]:(self.rois[i][0]+self.roi_size)]
                
                
                
                row_num = len(img1_arr)
                col_num = len(img2_arr[0])
                
               
                
                if (self.use_dark):
                    
                    dark_filenames = sf.FindFileNames(self.dark_folder)
                    
                    dark_arr3D = np.zeros((len(dark_filenames), row_num, col_num))
                    for i in range(len(dark_filenames)):
                        cur_dark = Image.open(self.dark_folder + dark_filenames[i])
                        dark_arr3D[i] = np.asarray(cur_dark,dtype = float)[self.rois[i][1]:(self.rois[i][1]+self.roi_size),self.rois[i][0]:(self.rois[i][0]+self.roi_size)]
                    dark_avg = np.average(dark_arr3D, axis=0)
                    self.img1_nobkg = img1_arr - dark_avg
                    self.img2_nobkg = img2_arr - dark_avg
                    
                else:
                    self.img1_nobkg = img1_arr
                    self.img2_nobkg = img2_arr
                    
        
            #calculate (auto/cross)correlation   
                self.res_corr = signal.correlate(self.img1_nobkg, self.img2_nobkg)
                        
                #normalize with respect to number of pairs of values that contribute to each correlation value:
                norm = np.ones((self.roi_size,self.roi_size)) 
                self.corr_norm = signal.correlate(norm, norm)
                self.res_corr /= self.corr_norm
                    
                #normalize with respect to average intensity, the way a (time) 
                #autocorrelation function is usually normalized
                self.i1ave = self.img1_nobkg.mean()
                self.i2ave = self.img2_nobkg.mean()
                self.I_avg.append(self.img1_nobkg.mean())
                self.res_corr /= (self.i1ave*self.i2ave)
                self.res_corr -= 1.
              
                        
                tstamp = time.asctime( time.localtime(time.time()) )
                tstamp = tstamp.replace(':','-')
                tstamp.replace(' ','_')
            
    
        
            #find peak position (pixel resolved)
                if self.autoc==True: #if autocorrelation: the 'real' peak must be at zero lag
                    indy = int(self.res_corr.shape[0]/2)
                    indx = int(self.res_corr.shape[1]/2)
                else:  #cross correlation: we don't know a priori where the peak is....
                    ind = np.unravel_index(np.argmax(self.res_corr, axis=None), self.res_corr.shape)
                    indy = ind[0]
                    indx = ind[1]
                
                #self.value_peak_correlation = self.res_corr[ indy, indx ]
                
                if axis == 0:
                    
                    self.ax = 0
                
                    self.cut_x = self.res_corr[indy]
                    self.cut_y = self.res_corr[:,indx]
                    self.CUTX.append(self.res_corr[indy])
                    self.peaks,_ = signal.find_peaks(self.cut_x,height=0.01)
                    
                    
                    length = len(self.peaks)
        
                    # Ensure the array length is odd
                    if length % 2 == 0:
                        raise ValueError("The array length must be odd")
                    
                    # Calculate the center index
                    center_index = length // 2
                    
                    center_index
                    
                    
                    down = self.peaks[center_index-1]
                    up = self.peaks[center_index+1]
                    
                    self.down.append(down)
                    self.up.append(up)
                    
                    a = sf.find_local_minima(self.cut_x[down:up])
                    
                    
                    
                    self.baseline.append(self.cut_x[down:up][a[0]])
                    
                    self.CUTY.append(np.asarray(self.cut_x)-np.asarray(self.cut_x[down:up][a[0]]))
                    
                    self.h_0.append( np.asarray(self.cut_x[self.peaks])[center_index]-np.asarray(self.cut_x[down:up][a[0]]) )
                    self.h.append(np.asarray(self.cut_x[self.peaks])[center_index-1]-np.asarray(self.cut_x[down:up][a[0]]))
                    self.rho.append((np.asarray(self.cut_x[self.peaks])[center_index-1]-np.asarray(self.cut_x[down:up][a[0]]))/(np.asarray(self.cut_x[self.peaks])[center_index]-np.asarray(self.cut_x[down:up][a[0]])))
                    self.Delta_x.append(self.peaks[center_index] - self.peaks[center_index-1])
                    self.Delta_y.append(0)
                    
                    
                else:
                    self.ax = 1
                    self.cut_x = self.res_corr[indy]
                    self.cut_y = self.res_corr[:,indx]
                    self.peaks,_ = signal.find_peaks(self.cut_y,height=0.05)
                    
                    
                    length = len(self.peaks)
        
                    # Ensure the array length is odd
                    if length % 2 == 0:
                        raise ValueError("The array length must be odd")
                    
                    # Calculate the center index
                    center_index = length // 2
                    
                    
                    try:
                        down = self.peaks[center_index-1]
                        up = self.peaks[center_index+1]
                        
                        self.down.append(down)
                        self.up.append(up)
                        
                        a = sf.find_local_minima(self.cut_y[down:up])
                        
                        
                        
                        #elf.baseline.append(self.cut_y[down:up][a[0]])
                        
                        self.CUTY.append(np.asarray(self.cut_y)  )
                        
                        h_0 = np.asarray( (self.cut_y[self.peaks])[center_index] -np.asarray(self.cut_y[down:up][a[0]]))
                        h = np.asarray( (self.cut_y[self.peaks])[center_index-1] -np.asarray(self.cut_y[down:up][a[0]]))
                        
                        self.h_0.append(h_0 )
                        self.h.append( h  )
                        #self.rho.append(self.h[i]/self.h_0[i])
                        self.rho.append(h)
                        self.Delta_x.append(0)
                        self.Delta_y.append(self.peaks[center_index] - self.peaks[center_index-1])
                    except IndexError:
                        print('ciao')
                        down = self.peaks[center_index]
                        up = self.peaks[center_index]
                    
                    
                        self.CUTY.append(np.asarray(self.cut_y)  )
                    
                        h_0 = np.asarray( (self.cut_y[self.peaks])[center_index] )
                        h = np.asarray(np.mean(self.h) )
                    
                        self.h_0.append(h_0 )
                        self.h.append( h  )
                        #self.rho.append(self.h[i]/self.h_0[i])
                        self.rho.append(h)
                        self.Delta_x.append(0)
                        self.Delta_y.append(np.mean(self.Delta_y))
                    
    def plot_rois_with_image( self,z_values):

        image_width, image_height = self.img1.size
    
        fig, ax = plt.subplots()
    
        
        # Display the image with origin at top-left
        ax.imshow(self.img1, extent=[0, image_width, image_height, 0], zorder=0,cmap='gray', vmin=0, vmax=100)
        
        # Normalize z-values to [0, 1] for colormap mapping
        norm = plt.Normalize(vmin=np.min(z_values), vmax=np.max(z_values))
        
        # Iterate through ROIs and plot points
        for roi, z in zip(self.rois, z_values):
            topx, topy = roi
            center_x = topx + self.roi_size / 2
            center_y = topy + self.roi_size / 2
            ax.plot(center_x, center_y, marker='o', markersize=5, color=plt.cm.hot(norm(z)), alpha=0.5)
        
    
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='hot', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=1)
        cbar.set_label('Z Value')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Points Inside ROIs Colored by Z Value')
    
        plt.show()
                   
    
    
    def calculate_radial_average( self,z_values, center_point, num_bins, exclude_region):
        x_exclude, y_exclude, exclude_width, exclude_height = exclude_region
    
        distances = []
        filtered_z_values = []
    
        for roi, z in zip(self.rois, z_values):
            topx, topy = roi
            if not (topx + self.roi_size > x_exclude and topx < x_exclude + exclude_width and
                    topy + self.roi_size > y_exclude and topy < y_exclude + exclude_height):
                center_x = topx + self.roi_size / 2
                center_y = topy + self.roi_size / 2
                distance = np.sqrt((center_x - center_point[0]) ** 2 + (center_y - center_point[1]) ** 2)
                distances.append(distance)
                filtered_z_values.append(z)
        
        distances = np.array(distances)
        filtered_z_values = np.array(filtered_z_values)
        
        # Bin the distances
        max_distance = np.max(distances)
        bins = np.linspace(0, max_distance, num_bins)
        bin_indices = np.digitize(distances, bins)
        
        radial_avg = []
        for i in range(1, num_bins):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                avg_z = np.mean(filtered_z_values[bin_mask])
                radial_avg.append((bins[i], avg_z))
        
        return radial_avg
    

    
    def smooth_radial_average(self,radial_avg,interp_points,label):
        distances, avg_z_values = zip(*radial_avg)
        
        
        yhat = savgol_filter(avg_z_values, interp_points, 3) 
        
        
        plt.figure()
        
        
        
        plt.plot(distances, avg_z_values, marker='o')
        plt.plot(distances,yhat )
        
        plt.xlabel('Radial Distance')
        plt.ylabel(label)
        plt.title('Radial Average ')
        plt.grid(True)
        plt.show()
        
        return distances, avg_z_values,yhat
        
        
                