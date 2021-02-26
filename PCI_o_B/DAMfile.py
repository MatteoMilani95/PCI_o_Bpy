# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:49:13 2021

@author: Matteo
"""

import numpy as np
import matplotlib.pyplot as plt
import PCI_o_B
from PCI_o_B import CIfile as CI
from PCI_o_B import G2file as g2
from PCI_o_B import SharedFunctions as sf

class DAM(g2.G2):
    def __init__(self,FolderName,CI,nROI,tau):
        
        super().__init__(FolderName,CI,nROI,tau)
        self.n_intervals = 0
        self.tauDAM= []
        self.g2DAM = []
        self.g2varDAM = []

        
    def __str__(self):
        #write again this stuff
        str_res  = '\n|---------------|'
        str_res += '\n| CIbead class:    '
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
    
    def DAMCalculation(self,n_intervals):
        self.n_intervals = n_intervals
        
        l_intervals = int(len(self.CI[0]) / n_intervals )
        time_list = []
        
        for i in range(n_intervals):
            time_list.append(i*l_intervals)
        
        #calculation of the g2 for each roi for each interval
        for i in range(n_intervals-1):
            super().G2Calculation(time_list[i],time_list[i+1])
            self.g2DAM.append(self.g2)
            self.tauDAM.append(np.asarray(self.tau))
            self.g2varDAM.append(self.g2var)

            
            self.g2 = []
            self.g2var = []
            #self.tau = []
            
        super().G2Calculation(time_list[-1],len(self.CI[0]))
        self.g2DAM.append(self.g2)
        self.g2varDAM.append(self.g2var)
        self.tauDAM.append(np.asarray(self.tau))
        
        '''
        for i in range(n_intervals):
            self.tauDAM[i].tolist()
            print(type(self.tauDAM[i]))
            print(len(self.tauDAM[i]))
        '''
        

        
        return
    
    
    def DAMFitSingleDecaytime(self,variables,plot):
        
        self.decaytime1DAM = []
        self.decaytime1errDAM = []
        
        for i in range(self.n_intervals):
            print(i)
            self.g2 = []
            self.g2var = []
            self.tau = []
            
            
            self.g2 = self.g2DAM[i]
            self.g2var = self.g2varDAM[i]
            self.tau = self.tauDAM[i]
            
            super().FitSingleDecaytime(variables,plot=False)
            
            self.decaytime1DAM.append(self.decaytime1)
            self.decaytime1errDAM.append(self.decaytime1err)

            
            self.decaytime1 = []
            self.decaytime1err = []

        
        return
    
    
    def DAMFitStretchDecaytime(self,variables,plot):
        
        self.decaytime1DAM = []
        self.decaytime1errDAM = []
        
        for i in range(self.n_intervals):
            print(i)
            self.g2 = []
            self.g2var = []
            self.tau = []
            
            
            self.g2 = self.g2DAM[i]
            self.g2var = self.g2varDAM[i]
            self.tau = self.tauDAM[i]
            
            super().FitStretchDecaytime(variables,plot=False)
            
            self.decaytime1DAM.append(self.decaytime1)
            self.decaytime1errDAM.append(self.decaytime1err)

            
            self.decaytime1 = []
            self.decaytime1err = []

        
        return
    
    
    def DAMFitDoubleDecaytime(self,variables,plot):
        
        self.decaytime1DAM = []
        self.decaytime1errDAM = []
        self.decaytime2DAM = []
        self.decaytime2errDAM = []
        
        for i in range(self.n_intervals):
            print(i)
            self.g2 = []
            self.g2var = []
            self.tau = []
            
            
            self.g2 = self.g2DAM[i]
            self.g2var = self.g2varDAM[i]
            self.tau = self.tauDAM[i]
            
            super().FitDoubleDecaytime(variables,plot=False)
            
            self.decaytime1DAM.append(self.decaytime1)
            self.decaytime1errDAM.append(self.decaytime1err)
            self.decaytime2DAM.append(self.decaytime2)
            self.decaytime2errDAM.append(self.decaytime2err)
            
            self.decaytime1 = []
            self.decaytime1err = []
            self.decaytime2 = []
            self.decaytime2err = []
        
        return
    
    
    def DAMFitSingleStretchDecaytime(self,variables,plot):
        
        self.decaytime1DAM = []
        self.decaytime1errDAM = []
        self.decaytime2DAM = []
        self.decaytime2errDAM = []
        
        for i in range(self.n_intervals):
            print(i)
            self.g2 = []
            self.g2var = []
            self.tau = []
            
            
            self.g2 = self.g2DAM[i]
            self.g2var = self.g2varDAM[i]
            self.tau = self.tauDAM[i]
            
            super().FitSingleStretchDecaytime(variables,plot=False)
            
            self.decaytime1DAM.append(self.decaytime1)
            self.decaytime1errDAM.append(self.decaytime1err)
            self.decaytime2DAM.append(self.decaytime2)
            self.decaytime2errDAM.append(self.decaytime2err)
            
            self.decaytime1 = []
            self.decaytime1err = []
            self.decaytime2 = []
            self.decaytime2err = []
        
        return
    
    def DAMFitDoubleStretchDecaytime(self,variables,plot):
        
        self.decaytime1DAM = []
        self.decaytime1errDAM = []
        self.decaytime2DAM = []
        self.decaytime2errDAM = []
        
        for i in range(self.n_intervals):

            self.g2 = []
            self.g2var = []
            self.tau = []
            
            
            self.g2 = self.g2DAM[i]
            self.g2var = self.g2varDAM[i]
            self.tau = self.tauDAM[i]
            
            super().FitDoubleStretchDecaytime(variables,plot=False)
            
            self.decaytime1DAM.append(self.decaytime1)
            self.decaytime1errDAM.append(self.decaytime1err)
            self.decaytime2DAM.append(self.decaytime2)
            self.decaytime2errDAM.append(self.decaytime2err)
            
            self.decaytime1 = []
            self.decaytime1err = []
            self.decaytime2 = []
            self.decaytime2err = []
        
        return