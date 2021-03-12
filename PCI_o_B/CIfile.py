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
import re

from PCI_o_B import SharedFunctions as sf

class CIfile():
    
    def __init__(self):
        """Initialize ROIs
        
        Parameters
        ----------
        Filename: complete filename of the CI file
        
        
        """
        
        
        
        self.FileList = []
        self.ConfigFolder = 'C:\\Scattering_CCD\\ConfigFiles'
        self.FolderName = []
        self.Input_101 = []
        self.Timepulse = False
       
        # ROI specification...
        self.nROI = []
        self.ROIlist = []
        self.ROIfilelist = []
        self.hsize = 0
        self.vsize = 0
        self.nROI = 0
        self.ROIlist = []
        self.GlobalROIhsize = 0
        self.GlobalROIvsize = 0
        self.GlobalROItopx = 0
        self.GlobalROItopy = 0
        self.ROI_x_pos = []
        
        self.lag = 0
        self.CI = []
        self.tau = []
        self.qvetcros = []
        
        #
        self.filename = 'ROI'
        self.cI_file_digits = 4
        self.extension = 'cI.dat'
        
        self.g2 = []
        self.g2var = []
        self.g2decaytime = []
        
        
        self.magnification = 0
        
        #self.Input_101 =[]
    def __repr__(self): 
        return '<ROI: fn%s>' % (self.FolderName)
    
    def __str__(self):
        str_res  = '\n|---------------|'
        str_res += '\n| CI class:    '
        str_res += '\n|--------------------+--------------------|'
        str_res += '\n| filelist:        ' + str(self.ROIfilelist)
        str_res += '\n| folder:          ' + str(self.FolderName) 
        str_res += '\n| number of ROIs : ' + str(self.nROI) 
        str_res += '\n| ROIs size      : ' + str(self.GetROIsize())+ ' px'
        str_res += '\n| lag time :       ' + str(self.lag) 
        #str_res += '\n| Window of interest top : ' + str(self.GetWINDOWtop()) + ' px'
        str_res += '\n|--------------------+--------------------|'
        return str_res
    
        
    def GetLagtime(self):
        return self.lag
    
    def GetFoldername(self):
        return self.FolderName
    
    def GetCI(self,*argv):
        try:
            return self.CI[argv[0]-1]
        except IndexError: 
            return self.CI
        
    def GetROIlist(self):
        return self.ROIlist
    
    def GetTau(self):
        return self.tau
    
    def GetROIsize(self):
        size=[]
        size.append(self.hsize)
        size.append(self.vsize)
        return size
    
    def GetROInumber(self):
        return self.nROI
    
    def GetWINDOWsize(self):
        size=[]
        size.append(self.GlobalROIhsize)
        size.append(self.GlobalROIvsize)
        return size
    
    def GetWINDOWtop(self):
        top=[]
        top.append(self.GlobalROItopx)
        top.append(self.GlobalROItopy)
        return top
    
    
    
    
        

    
    def SetROIlist(self,window_top,window_size,ROI_size):
        if len(self.ROIlist) != 0:
            print('WARNING: ROIlist already set, using this function you are changing ROIs. They won t be anymore the one of 101 file')
        
        self.hsize = ROI_size[0]
        self.vsize = ROI_size[1]
        self.GlobalROIhsize = window_size[0]
        self.GlobalROIvsize = window_size[1]
        self.GlobalROItopx = window_top[0]
        self.GlobalROItopy = window_top[1]
        self.ROIlist = []
        n_ROI=[]
        spaces=[]
  
        for i in range(len(window_top)):
           n,r=divmod(window_size[i]/ROI_size[i], 1)
           n_ROI.append(n)
           spaces.append(r)
           
        self.nROI = int(n_ROI[0]*n_ROI[1])
        
        if n_ROI[0] == 0 :
            print('ROI horizontal size larger then the orizontal size of the image')
            return
        if n_ROI[1] == 0 :
            print('ROI vertical size larger then the vertical size of the image')
            return
        gap_x = int((window_size[0] - n_ROI[0]*ROI_size[0])/n_ROI[0]) 
        top_x=[]
    
        for i in range(int(n_ROI[0])):
            if spaces[0] == 0:
                if i == 0:
                    top_x.append(window_top[0])
                else:
                    top_x.append(window_top[0]+i*ROI_size[0])
            else:
                if i == 0:
                    top_x.append(window_top[0]+int(gap_x/2))
                else:
                    top_x.append(window_top[0]+int(gap_x/2)+i*ROI_size[0]+gap_x)
#this part of the code shuold be optimize but I'm lazy.....
        gap_y = int((window_size[1] - n_ROI[1]*ROI_size[1])/n_ROI[1])
        top_y=[]
    
        for i in range(int(n_ROI[1])):
            if spaces[1] == 0:
                if i == 0:
                    top_y.append(window_top[1])
                else:
                    top_y.append(window_top[1]+i*ROI_size[1])
            else:
                if i == 0:
                    top_y.append(window_top[1]+int(gap_y/2))
                else:
                    top_y.append(window_top[1]+int(gap_y/2)+i*ROI_size[1]+gap_y) 
        for j in range(len(top_y)):
            for i in range(len(top_x)):
                self.ROIlist.append(top_x[i])
                self.ROIlist.append(top_y[j])
                self.ROIlist.append(ROI_size[0])
                self.ROIlist.append(ROI_size[1])
   

        return
        
    def UploadInput_101_CalCI(self):
        #
        ROIliststr = []
        
        
        for i in range(len(self.ROIlist)):
            ROIliststr.append(str(self.ROIlist[i])+'\n')
        
        
        try:
            with open(self.ConfigFolder+'\\Input_101_CalcCI.dat') as fp:
                self.Input_101 = fp.readlines()
                fi = self.Input_101.index('** IMPORTANT: the top left pixel of the image has coordinates(1, 1)\n')
                si = self.Input_101.index('** intensity threshold \n')
                f2 = self.Input_101.index('** id number of the first ROI (will be used for the name of the cI output file(s))\n')
                f1 = self.Input_101.index('** # of ROIs for which the correlation function will be calculated\n')
                fp.close()
                
                self.Input_101[fi+1:si] = ROIliststr
                self.Input_101[f1+1:f2] = str(str(self.nROI)+'\n') 
                
                open(self.ConfigFolder+'\\Input_101_CalcCI.dat','w').close()
                
                with open(self.ConfigFolder+'\\Input_101_CalcCI.dat', 'w') as f:
                    for item in self.Input_101:
                        f.write("%s" % item)
                    
                f.close()
                            
        except FileNotFoundError:
            print('FileNotFoundError: no Input_101_CalcCI.dat in this directory!')
            return 
        
        
    def LoadInput_101_CalCI(self):
        #Loading the ROIs starting from the Input_101_CalcCI.dat, this file is supposed to be in the same folder of the ROI files
        #this function allows to obtain the ROI list regardless the ROI list
        # is generated or not with the method GenerateROIlist
        
        self.nROI = 0
        self.ROIlist = []
        
        try:
            with open(self.FolderName+'\\Input_101_CalcCI.dat') as fp:
                self.Input_101 = fp.readlines()
            
            for i in range(len(self.Input_101)):
                if self.Input_101[i] == '** IMPORTANT: the top left pixel of the image has coordinates(1, 1)\n':
                    j=i+1
                    while self.Input_101[j] != '** intensity threshold \n':
                        self.ROIlist.append(int(self.Input_101[j]))
                        j=j+1
            fp.close()
        except FileNotFoundError:
            print('FileNotFoundError: no Input_101_CalcCI.dat in this directory!')
            return
        
        self.nROI = int(len(self.ROIlist)/4)
        
        self.hsize = int(self.ROIlist[2])
        self.vsize = int(self.ROIlist[3])
        return
        
  
        
    def LoadCI(self, FolderName,lagtime,magnification,Timepulse = False):
        #This method automatically call the method LoadInput_101_CalCI(), 
        #and the load the CI files for each ROI
        
        self.FolderName = FolderName
        self.lag = lagtime
        self.magnification = magnification
        self.LoadInput_101_CalCI()
        
        if Timepulse == False:
            
            ROI_name_list=list()
            
            for i in range(self.nROI):
                ROI_name_list.append(str(1 + i).zfill(self.cI_file_digits))
                self.ROIfilelist.append(self.filename + ROI_name_list[i]+ self.extension) 
                self.FileList.append(self.FolderName + '\\' + self.filename + ROI_name_list[i] + self.extension)
                
            for i in range(self.nROI):
                self.CI.append(pd.read_csv(self.FileList[i], sep='\\t', engine='python'))
            
            
            # get the tau list starting from the lag time
            for i in range(len(self.CI[0].columns)):
                if self.CI[0].columns[i].startswith('d'):
                    for char in self.CI[0].columns[i].split('d'):
                        if char.isdigit():
                            self.tau.append(float(char)*self.lag)
            return
                            
        else:
            
            self.Timepulse = True
            self.TimepulseOraganization()
            
            return
        
        return
    
    
    def TimepulseOraganization(self):
        ROI_name_list=list()
        CIlong = []
        filenamelong = 'longtcI'
        extension = '.dat'
            
        for i in range(self.nROI):
            ROI_name_list.append(str(1 + i).zfill(self.cI_file_digits))
            self.ROIfilelist.append(filenamelong + ROI_name_list[i]+ self.extension) 
            self.FileList.append(self.FolderName + '\\' + filenamelong + ROI_name_list[i] + extension)
                
        for i in range(self.nROI):
            CIlong.append(pd.read_csv(self.FileList[i], sep='\\t', engine='python'))
            
        ROI_name_list=list()
        CIshort = []
        filenameshort = 'shorttcI'
        extension = '.dat'
            
        for i in range(self.nROI):
            ROI_name_list.append(str(1 + i).zfill(self.cI_file_digits))
            self.ROIfilelist.append(filenameshort + ROI_name_list[i]+ self.extension) 
            self.FileList.append(self.FolderName + '\\' + filenameshort + ROI_name_list[i] + extension)
                
        for i in range(self.nROI):
            CIshort.append(pd.read_csv(self.FileList[self.nROI + i], sep='\\t', engine='python'))
            
        CIall = []
        cibho = []
        for i in range(self.nROI):
            CIall.append(CIshort[i].merge(CIlong[i], how='right', on = 'tsec', suffixes=('short', '')))
            CIall[i].set_index(CIlong[i].index, inplace=True)
            CIall[i].sort_values(by=['tsec'], inplace=True)
            CIall[i].reset_index(drop=True, inplace=True)
            #CIall[i].drop(['Iave', 'd0ave'], axis=1)
            CIall[i].drop(columns=['Iave', 'd0ave'], inplace=True)
            
            col_name="d0"
            first_col = CIall[i].pop(col_name)
            CIall[i].insert(0, col_name, first_col)
            
            col_name="I"
            first_col = CIall[i].pop(col_name)
            CIall[i].insert(0, col_name, first_col)
            
            col_name="tsec"
            first_col = CIall[i].pop(col_name)
            CIall[i].insert(0, col_name, first_col)
            cibho.append(CIall[i])
            
            
        self.CI = cibho
        self.tau.append(0)
        
        for i in range(len(self.CI[0].columns)):
                if self.CI[0].columns[i].startswith('usec'):
                    for char in self.CI[0].columns[i].split('usec'):
                        if char.isdigit():
                            self.tau.append(float(char)*10**-6)
            
        for i in range(len(self.CI[0].columns)):
                if self.CI[0].columns[i].startswith('sec'):
                    for char in self.CI[0].columns[i].split('sec'):
                        try:
                            self.tau.append(float(char))
                        except ValueError:
                            a = 0 
            
                            
        
        
        return 
    
    def CIShow(self,which_ROI,n_cycle=0,time_per_pair=0):
        
        folder_CI_graphs = self.FolderName + '\\CI_graphs'
        
        try:
            os.mkdir(folder_CI_graphs)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        if self.Timepulse == False:
            plt.figure() 
            for i in range(len(self.CI[0].columns)):
                if self.CI[0].columns[i].startswith('d'):
                    self.CI[which_ROI-1].plot(y=self.CI[0].columns[i],marker='.')
            
        else:
            plt.figure() 
            plt.title('CI ROI'+str(which_ROI).zfill(4)) 
            for i in range(len(self.CI[0].columns)):
                if self.CI[0].columns[i].startswith('usec'):
                    
                    a = self.CI[which_ROI-1][self.CI[which_ROI-1].columns[i]].dropna()
                    t=[]
                    for j in range(len(a)):
                        t.append(n_cycle*j)
                     
                    plt.plot(t,a.tolist(),label=self.CI[which_ROI-1].columns[i])
                    plt.ylabel('CI ')
                    plt.xlabel('time [s]')
                    
            

                    
            for i in range(len(self.CI[0].columns)):
                if self.CI[0].columns[i].startswith('sec'):
                    t=[]
                    for j in range(len(self.CI[which_ROI-1][self.CI[which_ROI-1].columns[i]].tolist())):
                        t.append(time_per_pair*j)
                    #self.CI[which_ROI-1].plot(y=self.CI[which_ROI-1].columns[i],marker='.',linestyle = 'solid')
                    plt.plot(t,self.CI[which_ROI-1][self.CI[which_ROI-1].columns[i]].tolist(),label=self.CI[which_ROI-1].columns[i])
                    plt.ylabel('CI')
                    plt.xlabel('time [s]')
            
            plt.savefig(folder_CI_graphs+'\\CI_ROI'+str(which_ROI).zfill(4)+'.png', dpi=300)

        return
    
    


    
    
    
class CIbead(CIfile):
    def __init__(self,n1,n2,wavelength):
        super().__init__()
        self.Radius = 0
        self.indexrefbead = n1
        self.indexrefext = n2
        self.scatt_angle = []
        self.q_vector = []
        self.scatt_angle_exp = []
        self.Center = 0
        self.wavelength = wavelength
        self.indexrefbead = n1
        self.indexrefext = n2
        self.decaytime = []
        
    def __str__(self):
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
        
        
    def SetThetaScatt(self,Radius):
        #insert the value of the radius in mm and the center in pixel!! This function has to be modified
        
        self.Radius = Radius
        self.Center = self.ROIlist[0] + ( self.ROIlist[( self.nROI - 1 ) * 4] + self.ROIlist[( self.nROI - 1 ) * 4 + 2] - self.ROIlist[0] )/2
        self.scatt_angle = []
        self.ROI_x_pos = []

        #self.ROI_x_pos = []
        for i in range(self.nROI):
            self.ROI_x_pos.append((-self.ROIlist[i*4] - self.ROIlist[i*4+2]/2 + self.Center)*self.magnification)
            
        if self.Radius-self.Center>self.ROI_x_pos[0] :
            print('ciao') 
        #H= np.array(ROI_x_pos)
        h = []
        
        for i in range(len(self.ROI_x_pos)):
            inner_h,scattering_angle=sf.theta1_func(self.ROI_x_pos[i],Radius,self.indexrefbead,self.indexrefext)
            h.append(inner_h)
            self.scatt_angle.append(scattering_angle*360/(2*math.pi))
                
        return
    
    def SetqScatt(self,Radius):
        
        self.SetThetaScatt(Radius)
        
        q=4*np.pi*self.indexrefbead*np.sin(np.asarray(self.scatt_angle) / 2 * np.pi / 180 ) / (532*10**-9)
        
        self.q_vector = q.tolist()
                
        return
    
    def TauPlastic(self,R_i,Ev_Rate):
        
        if R_i  <self.Radius:
            print('final R smaller than initial one')
            return
        else:

            x_i = []
            Dr = []
            tau_plasit = []
            x_f = self.ROI_x_pos
            for i in range(len(self.ROI_x_pos)):
                x_i.append(x_f[i] * R_i /(self.Radius-0.000001) )
                Dr.append(np.abs( x_i[i] - x_f[i] ) / R_i * 200 * 10**-6)
                tau_plasit.append(Dr[i] / Ev_Rate)
        
                
            plt.figure()
            plt.plot(self.q_vector,Dr,'o',label='plastic')
            plt.plot(self.q_vector,2*np.pi/np.asarray(self.q_vector),'o',label='2*pi/q')
            plt.xlabel('q vector [m-1]')
            plt.ylabel('l[m]')
            plt.title('evaporation rate = '+ str(Ev_Rate)+' m/s')
            plt.legend(loc='lower left')
            plt.savefig(self.FolderName + '\\fit_graphs\\evplot'+'.png')
            plt.show()
        
        return Dr
    
    
    
    
        
      
        
