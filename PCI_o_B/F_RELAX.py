# -*- coding: utf-8 -*-
"""
Created on Thu May  4 17:31:35 2023

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


class FORCE_RELAXATION_DATA():
    
    def __init__(self):
        """Initialize ROIs
        
        Parameters
        ----------
        Filename: complete filename of the CI file
        
        
        """
        
        self.time = []
        self.d = []
        self.F_n = []
        self.phi = []
        self.radius = []
        self.strain = []
        
    def __repr__(self): 
        return '<ROI: fn%s>' % (self.path)
    
    def __str__(self):
        str_res  = '\n|---------------|'
        str_res += '\n| XRAY_DATA class: '
        str_res += '\n|--------------------+--------------------|'
        return str_res
    
    def load_force_relaxation(self,path,phi,R):
        
        
        
        data = pd.read_csv(path,skiprows=25,usecols=[1,4,5], decimal=",",encoding='UTF-16 LE', header=None,delimiter=r"\s+")
        
        
        time = data[1]
        F_n = data[5]
        distance = data[4]
        d = np.mean(distance)

        
        self.time.append(np.asarray(time))
        self.F_n.append(np.asarray(F_n))
        self.radius.append(R)
        self.phi.append(phi)
        self.d.append(d)
        self.strain.append((R-d)/R)
        return
    
    
    
    
    def plot_force_time(self,t_min,t_max):
        
        
        
        n = len(self.F_n)
        colors = pl.cm.inferno(np.linspace(0,1,n))

        
        fig = plt.figure()
        ax = plt.axes()
        
        for i in range(len(self.F_n)):
            ax.set_xscale("log")
            #ax.set_yscale("log")
            ax.plot(self.time[i],self.F_n[i],color=colors[i],label=r'$\epsilon$ = '+str(np.round(self.strain[i],2)))
            ax.set_ylabel(r'$F_N$ $[N]$',fontsize=15)
            ax.set_xlabel(r'$t$ $[s] $',fontsize=15)
            ax.legend()
            ax.set_xlim([t_min,t_max])
            
            
    def plot_normalized_force_time(self,t_min,t_max):
        
        
        
        n = len(self.F_n)
        colors = pl.cm.inferno(np.linspace(0,1,n))

        
        fig = plt.figure()
        ax = plt.axes()
        
        for i in range(len(self.F_n)):
            ax.set_xscale("log")
            #ax.set_yscale("log")
            ax.plot(self.time_normalized[i],self.F_n_normalized[i],color=colors[i],label=r'$\epsilon$ = '+str(np.round(self.strain[i],2)))
            ax.set_ylabel(r'$F_N$ $[N]$',fontsize=15)
            ax.set_xlabel(r'$t$ $[s] $',fontsize=15)
            ax.legend()
            ax.set_xlim([t_min,t_max])

            
        
        
        return
    
    
    
   
    
    def cut_tails(self,t_max,plot=True):
        
        start = []
        
        
        for i in range(len(self.time)):
            start.append(sf.find_nearest(self.time[i], t_max[i]))
        
        lst = []
        
        for i in range(len(self.time)):
            lst.append(list(range(start[i],len(self.time[i]))))
            
            if plot == True:
                plt.figure()
                ax = plt.axes()
                ax.semilogx(self.time[i],self.F_n[i],marker='.',linestyle='')
                ax.semilogx(self.time[i][lst[i]],self.F_n[i][lst[i]],marker='.',linestyle='',color='red',label='phi = '+ str(self.phi[i]) )
                ax.set_xlabel(r'$t$' + str(' ') + ' [s]',fontsize=18)
                
                ax.set_ylabel(r'$F_n$' + str(' ') + ' [N]',fontsize=18)
                ax.legend(loc='upper right')
            
        
            self.time[i] = np.delete(self.time[i],lst[i],0)
            self.F_n[i] = np.delete(self.F_n[i],lst[i],0)
        return
    
    
    def normalize_time_and_force(self,plot=True):
        
        self.time_normalized = []
        self.F_n_normalized = []
        
        for i in range(len(self.F_n)):
            self.F_n_normalized.append((self.F_n[i]-self.F_n[i][-1])/self.F_n[i][0])
            self.time_normalized.append(self.time[i]-self.time[i][0]+0.01)
        
        if plot == True:
            n = len(self.F_n)
            colors = pl.cm.winter(np.linspace(0,1,n))
    
            
            fig = plt.figure()
            ax = plt.axes()
            
            for i in range(len(self.F_n)):
                ax.set_xscale("log")
                
                ax.plot(self.time_normalized[i],self.F_n_normalized[i],color=colors[i],label=r'$\epsilon$ = '+str(np.round(self.strain[i],2)))
                ax.set_ylabel(r'$\frac{F_N}{F_max}$ $[]$',fontsize=15)
                ax.set_xlabel(r'$t-t_0$ $[]$ ',fontsize=15)
                ax.legend()
                
        
        return

    