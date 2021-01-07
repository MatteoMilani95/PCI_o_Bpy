# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:02:48 2020

@author: Matteo
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from pynverse import inversefunc
from IPython import get_ipython
get_ipython().magic('reset -sf')
import pandas as pd
from scipy.optimize import leastsq, least_squares, curve_fit
import os

from PCI_o_B import SharedFunctions as sf


class G2():
    def __init__(self,FolderName,CI,nROI,tau):
        super().__init__()
        
        self.FolderName = FolderName
        self.CI = CI
        self.nROI = nROI
        self.g2 = []
        self.g2var = []
        self.tau = tau
        self.scatt_angle = []
        self.scatt_angle_exp = []
        self.Center = 0
        self.decaytime1 = []
        self.decaytime1err = []
        self.decaytime2 = []
        self.decaytime2err = []
        
    def __str__(self):
        str_res  = '\n|---------------|'
        str_res += '\n| G2 class:    '
        str_res += '\n|--------------------+--------------------|'
        str_res += '\n| filelist             : ' + str(self.ROIfilelist)
        str_res += '\n| folder               : ' + str(self.FolderName) 
        str_res += '\n| number of ROIs       : ' + str(self.nROI) 
        str_res += '\n| ROIs size            : ' + str(self.GetROIsize()) + ' px'
        str_res += '\n| lag time             : ' + str(self.lag)
        str_res += '\n| x for theta(x)= 90°  : ' + str(self.Center) + 'px'
        str_res += '\n| Radius bead          : ' + str(self.Center) +'px'
        #str_res += '\n| Window of interest top : ' + str(self.GetWINDOWtop()) + ' px'
        str_res += '\n|--------------------+--------------------|'
        return str_res
    
    
    def G2Calculatino(self):
        
        g2=[]
        var=[]
        for i in range(self.nROI):
            g2.append(self.CI[i].mean(axis = 0)) 
            var.append(self.CI[i].var(axis = 0)) 
        remove_nan = []
        remove_nan_v = []
        
        for i in range(self.nROI):
            nan_elems = g2[i].isnull()
            nan_elems_v = var[i].isnull()
            remove_nan.append(g2[i][~nan_elems])
            remove_nan_v.append(var[i][~nan_elems_v])
            
        
        for i in range(self.nROI):
            self.g2.append(remove_nan[i][2:])
            #self.g2 = [x for x in self.g2 if str(x) != 'nan']
            self.g2var.append(remove_nan_v[i][2:])
       
            
        while len(self.tau)>len(self.g2[0]):
            self.tau.pop()
            
        return
    
   
        
    def fitG2(self,function,variables):
        
        outparam = []
        
        for i in range(self.nROI):
            outparam.append(curve_fit(function,  np.asarray(self.tau), np.asarray(self.g2[i]), variables, np.asarray(self.g2var[i]) ))

        return outparam
    
    
    def FindSingleDecaytime(self,func,variables,plot):
        
        fitted_curve = []
        
        outparam = self.fitG2(func,variables)
        
        folder_fit_graphs = self.FolderName + '\\fit_graphs'
        
        try:
            os.mkdir(folder_fit_graphs)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        for i in range(self.nROI):
             fitted_curve.append(func( np.asarray(self.tau), outparam[i][0][0], outparam[i][0][1], outparam[i][0][2]))
             
    
        for i in range(self.nROI):
            self.decaytime1.append(outparam[i][0][1])
            self.decaytime1err.append(2*np.sqrt(outparam[i][1][1][1]))
        
        goodness_fit = [] 
        for i in range(self.nROI):
            goodness_fit.append( np.sum( ( ( np.asarray(fitted_curve[i])-np.asarray(self.g2[i]))  )**2 / np.asarray(fitted_curve[i]) ) )
    
        if plot == True:
            for i in range(self.nROI):
                plt.figure() 
                plt.xscale('log')
                plt.errorbar(self.tau,self.g2[i],yerr=self.g2var[i],fmt='o',label='chi = ' + str(goodness_fit[i]) + '\n' + 'baseline = ' + str(outparam[i][0][2]) + '\n' + 'decaytime = ' + str(outparam[i][0][1]))
                plt.plot(self.tau,fitted_curve[i],'-.')
                plt.xlabel('tau  [s]')
                plt.ylabel('g2-1')
                plt.title('g2_ROI'+str(i+1).zfill(4))
                plt.legend(loc='lower left')
                #plt.grid(True)
                plt.savefig(folder_fit_graphs+'\\g2_ROI'+str(i+1).zfill(4)+'.png')
        else:
            return
        
        return
    
    def FindDoubleDecaytime(self,func,variables,plot):
        
        fitted_curve = []
        
        outparam = self.fitG2(func,variables)
        
        folder_fit_graphs = self.FolderName + '\\fit_graphs'
        
        try:
            os.mkdir(folder_fit_graphs)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        for i in range(self.nROI):
             fitted_curve.append(func( np.asarray(self.tau), outparam[i][0][0], outparam[i][0][1], outparam[i][0][2], outparam[i][0][3], outparam[i][0][4]))
             
    
        for i in range(self.nROI):
            self.decaytime1.append(outparam[i][0][1])
            self.decaytime1err.append(2*np.sqrt(outparam[i][1][1][1]))
            self.decaytime2.append(outparam[i][0][3])
            self.decaytime2err.append(2*np.sqrt(outparam[i][1][3][3]))
        
        goodness_fit = [] 
        for i in range(self.nROI):
            goodness_fit.append( np.sum( ( ( np.asarray(fitted_curve[i])-np.asarray(self.g2[i]))  )**2 / np.asarray(fitted_curve[i]) ) )
    
        if plot == True:
            for i in range(self.nROI):
                plt.figure() 
                plt.xscale('log')
                plt.errorbar(self.tau,self.g2[i],yerr=self.g2var[i],fmt='o',label='chi = ' + str(goodness_fit[i]) + '\n' + 'baseline = ')
                plt.plot(self.tau,fitted_curve[i],'-.')
                plt.xlabel('tau  [s]')
                plt.ylabel('g2-1')
                plt.title('g2_ROI'+str(i+1).zfill(4))
                plt.legend(loc='upper right')
                #plt.grid(True)
                plt.savefig(folder_fit_graphs+'\\g2_ROI'+str(i+1).zfill(4)+'.png')
        else:
            return
        
        return
    

    
    def G2Show(self,which_ROI):
        
        plt.figure() 
        plt.xscale('log')
        plt.errorbar(self.tau,self.g2[which_ROI-1],yerr=self.g2var[which_ROI-1],fmt='o')
        plt.xlabel('tau  [s]')
        plt.ylabel('g2-1')
        plt.title('g2_ROI'+str(which_ROI).zfill(4))

        return
    
    def G2CutBaseLine(self,nPoints):
        
        cut = []
        for i in range(nPoints):
            cut.append(self.g2[0].index[-i-1])
        
        for i in range(self.nROI):
            self.g2[i].drop(cut,inplace=True)
            self.g2var[i].drop(cut,inplace=True)
        
        x=self.tau[:len(self.g2[0])]
        self.tau=[]
        self.tau=x
        
        return cut
    
    def G2Normalize(self):
        for i in range(self.nROI):
            self.g2[i] = self.g2[i] / self.g2[i][1]
        
        
        return
        
