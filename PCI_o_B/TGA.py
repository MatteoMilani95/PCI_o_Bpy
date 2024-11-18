# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:11:30 2023

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
import sys
from scipy import interpolate
import numpy.core.defchararray as np_f



class TGA_DATA():
    
    def __init__(self):
        """Initialize ROIs
        
        Parameters
        ----------
        Filename: complete filename of the CI file
        
        
        """
        
      
        
       
        
        
        #self.Input_101 =[]
    def __repr__(self): 
        return '<ROI: fn%s>' % (self.path)
    
    def __str__(self):
        str_res  = '\n|---------------|'
        str_res += '\n| TGA_DATA class: '
        str_res += '\n|--------------------+--------------------|'
        return str_res
    

    
    def load_data(self,path):
        
        data = pd.read_csv(path,skiprows=2,skipfooter=3,delimiter='\s+',engine='python')

        t=data.iloc[:, 1].values
        c = np.asarray(list(map(lambda s: s.replace(',' , '.'), t)))
        self.time = c.astype(np.float)
        
        T=data.iloc[:, 3].values
        a = np.asarray(list(map(lambda s: s.replace(',' , '.'), T)))
        self.temp = a.astype(np.float)
        
        M=data.iloc[:, 2].values
        b = np.asarray(list(map(lambda s: s.replace(',' , '.'), M)))
        self.mass = b.astype(np.float)
        
        self.relative_mass = self.mass/np.max(self.mass)
        

        
        
        return
    
    def loss_percentage(self,index):
        
       
        print('wheight loss = '+ str((1-self.relative_mass[index])*100) + ' %')
        
        plt.figure()
        ax = plt.axes()
        ax.plot(self.time,self.relative_mass)
        ax.plot(self.time[index],self.relative_mass[index],marker='o',color='red')
        
        ax.set_xlabel(r'$t$ [s]',fontsize=20)
        ax.set_ylabel(r'$m/m_0$ [-]',fontsize=20)
        
        return