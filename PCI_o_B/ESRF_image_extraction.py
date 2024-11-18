# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:28:17 2024

@author: Matteo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:23:33 2024

@author: Matteo
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as integrate
from scipy import interpolate
from numpy import nan
from scipy.special import gamma, factorial
import openpyxl
import matplotlib.pylab as pl
from matplotlib.widgets import Cursor
from PCI_o_B import SharedFunctions as sf
import os
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import h5py, hdf5plugin


class XRAY_DATA():
    
    def __init__(self):
        """
        processed Parameters
        
        
        """
        
        self.q = []
        self.I_q = []
        self.trm =[]
        self.epoch = []
        
        """
        raw Parameters
        
        
        """
        
        
        self.images = []
        
        
        
        self.df  = []
        self.phi = []
        self.radius = []
        self.lc = []
        self.alpha = []
        self.volume = []
        self.absolute_normalization = []
       
        
        
        #self.Input_101 =[]
    def __repr__(self): 
        return '<ROI: fn%s>' % (self.path)
    
    def __str__(self):
        str_res  = '\n|---------------|'
        str_res += '\n| XRAY_DATA class: '
        str_res += '\n|--------------------+--------------------|'
        return str_res
    
    

    
    
    def load_and_save_images(folder_path):
        
        filelist = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if 'cam' in f ]
        
        folderout = folder_path + r'/extracted_images'
        
        try:
            os.mkdir(folderout)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        
        
        for i in range(len(filelist)):
            
            df = np.array(h5py.File(filelist[i])['entry_0000/ESRF-ID02/cam/data'])
            
            
            
            
            plt.imsave(folderout+'/first_image_dataset_'+ str( i+1 ).zfill(5)+'.png', df[0,410:560,420:900])
        
        
        
        return

    
    
def list_cam_files(folder_path):
    """
    List all files in the specified folder that contain 'cam' in the name 
    and end with '_ave.h5'.
        
    Parameters:
    folder_path (str): The path to the folder to search in.
        
    Returns:
    list: A list of filenames that contain 'cam' and end with '_ave.h5'.
    """
    h5_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if 'cam' in f ]
    
    outfold = folder_path + r'/out'
    
    try:
        os.mkdir(outfold)
    except FileExistsError:
        print('directory already existing, graphs will be uploaded')
            
            
            
    return h5_files

