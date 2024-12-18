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
from scipy import stats
from PCI_o_B import SharedFunctions as sf
from datetime import datetime
import shutil
import matplotlib.pylab as pl

class CI():
    
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
        self.Timepulse2 = False
        self.Timepulse3 = False
       
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
        self.Iav = []
        
        #
        self.filename = 'ROI'
        self.cI_file_digits = 4
        self.extension = 'cI.dat'
        
        self.g2 = []
        self.g2var = []
        self.g2decaytime = []
        
        
        
        #self.Input_101 =[]
    def __repr__(self): 
        return '<ROI: fn%s>' % (self.FolderName)
    
    def __str__(self):
        str_res  = '\n|---------------|'
        str_res += '\n| CI class:    '
        str_res += '\n|--------------------+--------------------|'
        str_res += '\n| folder         : ' + str(self.FolderName) 
        str_res += '\n| number of ROIs : ' + str(self.nROI) 
        str_res += '\n| ROIs size      : ' + str(self.GetROIsize())+ ' px'
        str_res += '\n| lag time       : ' +"{:.4f}".format(self.lag ) + ' s'
        str_res += '\n| timepulse      : ' +str(self.Timepulse2)
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
    
    
    
    
        

    
    def SetROIlist(self,window_top,window_size,ROI_size,Overlap=False):
        if len(self.ROIlist) != 0:
            print('WARNING: ROIlist already set, using this function you are changing ROIs. They won t be anymore the one of 101 file')
        
        self.hsize = ROI_size[0]
        self.vsize = ROI_size[1]
        self.GlobalROIhsize = window_size[0]
        self.GlobalROIvsize = window_size[1]
        self.GlobalROItopx = window_top[0]
        self.GlobalROItopy = window_top[1]
        self.ROIlist = []
        
        if Overlap == False:
        
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
       
    

        
        else:
            
            summ = 0;
            top_x=[]
            top_y=[]
            self.nROI = 0
            while self.GlobalROItopx + summ + self.hsize < self.GlobalROItopx +self.GlobalROIhsize :
                top_x.append(self.GlobalROItopx+summ)
                top_y.append(self.GlobalROItopy)
                summ = summ + 50
                self.nROI = self.nROI + 1
                
            for j in range(len(top_y)):
                self.ROIlist.append(top_x[j])
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
        
  
        
    def LoadCI(self, FolderName,lagtime,Normalization = False,Timepulse = False):
        #This method automatically call the method LoadInput_101_CalCI(), 
        #and the load the CI files for each ROI
        
        self.FolderName = FolderName
        self.lag = lagtime
        self.LoadInput_101_CalCI()
        
        if Timepulse == False:
            
            ROI_name_list=list()
            
            for i in range(self.nROI):
                ROI_name_list.append(str(1 + i).zfill(self.cI_file_digits))
                self.ROIfilelist.append(self.filename + ROI_name_list[i]+ self.extension) 
                self.FileList.append(self.FolderName + '\\' + self.filename + ROI_name_list[i] + self.extension)
                
            for i in range(self.nROI):
                self.CI.append(pd.read_csv(self.FileList[i], sep='\\t', engine='python'))
            
            
            
            if Normalization == True:
                self.NoiseNormalization()
                
                
            # get the tau list starting from the lag time
            for i in range(len(self.CI[0].columns)):
                if self.CI[0].columns[i].startswith('d'):
                    for char in self.CI[0].columns[i].split('d'):
                        if char.isdigit():
                            self.tau.append(float(char)*self.lag)
            return
                            
        else:
            
            self.Timepulse = True
            print('deprecated (2021/04/16) use the function TimepulseOraganization instead')
            self.TimepulseOraganization()
            
            return
        
        return
    
    
    def LoadCI_correction(self, FolderName,lagtime,Normalization = False,Timepulse = False):
        #This method automatically call the method LoadInput_101_CalCI(), 
        #and the load the CI files for each ROI
        
        self.FolderName = FolderName
        self.lag = lagtime
        self.LoadInput_101_CalCI()
        
        if Timepulse == False:
            
            ROI_name_list=list()
            
            for i in range(self.nROI):
                ROI_name_list.append(str(1 + i).zfill(self.cI_file_digits))
                self.ROIfilelist.append(self.filename + ROI_name_list[i]+ self.extension) 
                self.FileList.append(self.FolderName + '\\' + self.filename + ROI_name_list[i] + self.extension)
                
            for i in range(self.nROI):
                self.CI.append(pd.read_csv(self.FileList[i], sep='\\t', engine='python'))
            
            
            
            if Normalization == True:
                self.NoiseNormalization()
                
                
            # get the tau list starting from the lag time
            for i in range(len(self.CI[0].columns)):
                if self.CI[0].columns[i].startswith('d'):
                    for char in self.CI[0].columns[i].split('d'):
                        if char.isdigit():
                            self.tau.append((float(char)-1)/2*self.lag)
            return
                            
        else:
            
            self.Timepulse = True
            print('deprecated (2021/04/16) use the function TimepulseOraganization instead')
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
    
    
    
    
    def LoadTimePulseCI(self, FolderName,Normalization = False):
        #This method automatically call the method LoadInput_101_CalCI(), 
        #and the load the CI files for each ROI
        
        self.FolderName = FolderName
        self.LoadInput_101_CalCI()
        
        #get list of time delays within a cycle
        
        
        pulse_time,n_pulses,cycle_dur = self.TimePulseLoad()
        self.lag = pulse_time[1]
        
        ROI_name_list=[]
            
        for i in range(self.nROI):
            ROI_name_list.append(str(1 + i).zfill(self.cI_file_digits))
            self.ROIfilelist.append(self.filename + ROI_name_list[i]+ self.extension) 
            self.FileList.append(self.FolderName + '\\' + self.filename + ROI_name_list[i] + self.extension)
            
        
                
        for i in range(self.nROI):
            self.CI.append(pd.read_csv(self.FileList[i], sep='\\t', engine='python'))            
            
        self.Timepulse2 = True   
        
        if Normalization == True:
            self.NoiseNormalization()
            
            
            
        delays = np.asarray(self.CI[0].columns[2:],dtype = str)
        for i in range(delays.size): delays[i] = delays[i].replace('d','')  
        delays = delays.astype(int)
        ndelays = delays.size

                        
        #time at which each image was taken (in sec, t=0 at the beginning of the cI file)
        ntimes = self.CI[0]['n'].size
        time_im = np.zeros(ntimes,dtype = np.float64) 
        time_im[0] =  pulse_time[0]
        for j in range(0,time_im.size):
            time_im[j] = cycle_dur*(j//n_pulses) + pulse_time[j%n_pulses]  
            
            
        #time delay between all pairs of images for which cI has been calculated
        tau_true = np.ones((ntimes,ndelays),dtype = np.float64)*np.nan
        for r in range(ntimes):
            for c in range(ndelays):
                r2 = r+delays[c]
                if r2 < ntimes: tau_true[r,c] = time_im[r2]-time_im[r]     
                
                
        tau_true = np.round(tau_true,6) #the time resolution is 1E-6 sec....

        #get a sorted array with all unique delays, excluding nan and inf
        a = np.sort(np.unique(tau_true))
        a = a[np.isfinite(a)]
        # "consolidate" list of delays, by grouping delays whose ratio is between 1
        # and rel_diff
        rel_diff = 1.05
        #define bins to which all delays will be assigned. To avoid problems with roundoff, we slightly shift all bin edges to the left
        epsilon = 1E-6
        bins = [a[0]-epsilon,a[1]-epsilon]
        print(bins)#define the first bin so as it contains a[0]
        pb = a[1]
        for j in range(2,a.size):
            if a[j] >= rel_diff*pb:
                bins.append(a[j]-epsilon)
                pb = a[j]
        #get time delay corresponding to each bin: average of time delays that belong to
        # that bin
        tau_mean, bin_edges, binnum = stats.binned_statistic(a,a,statistic = 'mean', bins=bins)
            
        self.tau = list(tau_mean)
        
        col_names = ['tsec','n','Iav']
        for t in tau_mean:
            col_names.append(format(t,'.3e')+' s')

            
            # "consolidate" cIs, i.e. for each t average them over delays tau that fall in
            # the same bin. Store in pandas dataframe and output to file consolidated cIs
        for i in range(self.nROI):
            now = datetime.now()
            print("time =", now)
            print('Calculating cIs for all available time delays (in sec), for ROI ' + str(i+1) + ' over ' +str(self.nROI))
            cIcons = pd.DataFrame(index=range(ntimes),columns=col_names)
            cIcons['tsec'] = time_im
            cIcons['n'] = self.CI[i]['n']
            cIcons['Iav'] = self.CI[i]['Iav']
            binning = []
            binindex = []
            for j in range(time_im.size):
                
                cI = np.asarray(self.CI[i].iloc[j,2:])
                good = ~np.isnan(tau_true[j]) & ~np.isnan(cI)
                prova = []
                if (cI[good].size>0):
                    now = datetime.now()
                    #print("now_befor_av =", now)
                    cImean, bin_edges, binnum2 = stats.binned_statistic(tau_true[j][good],\
                                cI[good], statistic = 'mean', bins=bins)
                    now = datetime.now()
                    #print("now_after_av =", now)
                    now = datetime.now()
                    #print("now_befor_iloc =", now)
                    cIcons.iloc[j,3:] = cImean
                    now = datetime.now()
                    #print("now_after_iloc =", now)
                    
                    
                    
                    
                    binning.append(bin_edges)
                    binindex.append(binnum2)
        
            
            self.CI[i] = []
            self.CI[i] = cIcons
            
        
        for i in range(self.nROI): 
            
            self.Iav.append(self.CI[i]['Iav'])
            self.CI[i].drop(['Iav'], axis=1,inplace=True)
                              
                        
        return
    
    def consolidate(self, in2darray,il,ih):
        """
        utility function to 'consolidate' (i.e. average over row-dependent groups
        of columns) a 2d array. The input array may contain nan's. Sets to nan the
        output when no valid data are available
    
        Parameters
        ----------
        in2darray : TYPE 2d numpy array, any type (recommended: float64)
            DESCRIPTION. data array, shape(Nr,Nc)
        il : TYPE int, 2d array, shape(Nr,Nbins)
            DESCRIPTION. lower index for defining the (row-dependent) range of 
            columns over which in2darray has to be averaged
        ih : TYPE int, 2d array, shape(Nr,Nbins)
            DESCRIPTION. higher index for defining the (row-dependent) range of 
            columns over which in2darray has to be averaged
    
        Returns
        -------
        cons : TYPE 2d numpy array, float64, size Nr rows and Nbins columns
            DESCRIPTION. The average, row-by row, of in2darray. Averaging is done
            over groups of columns, specified by il and ih (inclusive):
            cons[r,b] = np.nanmean(in2darray[r,il[b]:ih[b]+1],axis=1)    
    
        """
        verb = False #to enable/disable various check print
        if verb: print('in2darray.shape:',in2darray.shape)
        Nr = in2darray.shape[0]
        Nc = in2darray.shape[1]
        Nbins = il.shape[1]
        if verb: print('Nbins',Nbins)
        if il.shape != ih.shape:
            raise NameError('consolidate(): il and ih must have the same shape')
        if il.shape[0] != Nr:
            raise NameError('consolidate(): il, ih, in2array must have the same'+\
                            ' number of rows')
        
        cons = np.ones((Nr,Nbins),dtype = np.float64)
        cons *= np.nan
        for r in range(Nr):
            for b in range(Nbins):
                if il[r,b] >= 0: #il is set to -1 for those bins where no valid data
                                #are available
                    h = min(ih[r,b]+1,Nc)
                    if verb: print(r,il[r,b],h)
                    cons[r,b] = np.nanmean(in2darray[r,il[r,b]:h])
                    #note that we average up to column ih[r,b] INCLUDED
        return cons
    
    
    
    def build_indexes(self, tau_true,binl,binh):
        """
        Given a 2d array of time delays tau_true and a list of (time delay) 
        bins, calculates, for each row of tau_true, the lowest and highest column 
        index such that tau_true delay belongs to a given bin
    
        Parameters
        ----------
        tau_true : TYPE numpy 2d array, shape (Nr,Nc), expected dtype: float
            DESCRIPTION. 2d array of time delays between all pairs of images. rows
            correspond to timle of first image, columns to delays between first and
            second image
        binl : TYPE list of length Nbins
            DESCRIPTION. lower edges of the delay time bins to be used.
        binh : TYPE list of length Nbins
            DESCRIPTION. higher edges of the delay time bins to be used.
    
        Returns
        -------
        indexl : TYPE numpy 2d array, int, shape(Nr,Nbins)
            DESCRIPTION. lower index, see below
        indexh : TYPE numpy 2d array, int, shape(Nr,Nbins)
            DESCRIPTION. higher index, see below
        
        Note: tau_true[r,indexl[r,b]:indexh[r,b]+1] is the set of time delays that
        belong to the b-th bin, for row r (i.e. for the r-th time of the first 
                                           image)
    
        """
        Nr = tau_true.shape[0]
        Nbins = len(binl)
        if len(binh) != Nbins:
            raise NameError('build_indexes(): binl and binh must have the same size')
    
        indexl = -np.ones((Nr,Nbins),dtype = int)
        indexh = -np.ones((Nr,Nbins),dtype = int)
        
        for r in range(Nr):
            for b in range(Nbins):
                # w = np.where( (tau_true[r,np.isfinite(tau_true[r])]>=binl[b]) & \
                #               (tau_true[r,np.isfinite(tau_true[r])]< binh[b]))                
                w = np.where( (tau_true[r]>=binl[b]) & \
                              (tau_true[r]< binh[b]))                
    
                if w[0].size > 0:
                    indexl[r,b] = w[0][0]
                    indexh[r,b] = w[0][-1]
        
        return indexl,indexh
    
    def LoadConsolidate(self, folderin,ROIlist,rel_diff,normalization=False): 
        
        self.FolderName = folderin
        self.Timepulse3 = True
        
        if folderin[-1] != '/' and folderin[-1] != '\\': folderin += '/'

        #Note: we assume that all the ROIs have been processed with the same set of 
        #parameters (same set of images, same set of time delays etc.)
        
        
        self.nROI = len(ROIlist)
        
        for ROInum in ROIlist:
            print('Processing data for ROI n.%d' % ROInum)
            
            #check if the npz file with the info for consolidating data exists
            in_npzfile = folderin + 'consolidate_info.npz'
            calc_cons = True
            try:
                npzfile = np.load(in_npzfile)
                indexl = npzfile['indexl']
                indexh = npzfile['indexh']
                tau_cons = npzfile['tau_cons']
                tau_mean = npzfile['tau_mean']
                time_im = npzfile['time_im']
            except:
                calc_cons = True
        
            
            filein = folderin + 'ROI' + str(ROInum).zfill(4) + 'cI.dat'
            cIraw = pd.read_csv(filein, sep="\t") #read cI file (note: fills with NaN missing data)
            
            if normalization == True:
                
                
                for i in range(len(cIraw.columns[3:])): 
                    for j in range(cIraw.iloc[:,3+i].count()):
                        cIraw.iloc[j,3+i] = cIraw.iloc[j,3+i] / np.sqrt( cIraw.iloc[j,2] * cIraw.iloc[j+i+1,2] )

        
                
            
            ntimes = cIraw['n'].size
            if calc_cons:
                #get list of delays (in number of images):
                delays = np.asarray(cIraw.columns[2:],dtype = str)
                for i in range(delays.size): delays[i] = delays[i].replace('d','')  
                delays = delays.astype(int)
                ndelays = delays.size
            
                #get list of time delays within a cycle
                #TO DO: find time of each image in different ways (TimePulses, Guillaume's
                #file with image time, ImagesLog etc.) 
                filein = folderin + 'TimePulses.txt'
                cycle_data =  pd.read_csv(filein, sep="\t",header = None)
                n_pulses = int(cycle_data[0][0])
                pulse_time = np.asarray(cycle_data[0][1:n_pulses+1],dtype=np.float64)/1E6  #in sec
                cycle_dur = np.asarray(cycle_data[0][n_pulses+1],dtype=np.float64)/1E6  #in sec
                
                #time at which each image was taken (in sec, t=0 at the beginning of the cI file)
                
                time_im = np.zeros(ntimes,dtype = np.float64) 
                time_im[0] =  pulse_time[0]
                for j in range(0,time_im.size):
                   time_im[j] = cycle_dur*(j//n_pulses) + pulse_time[j%n_pulses]
                
                #time delay between all pairs of images for which cI has been calculated
                print('\ncalculating the time delays between all pairs of images...')
                tau_true = np.ones((ntimes,ndelays),dtype = np.float64)*np.nan
                for r in range(ntimes):
                    for c in range(ndelays):
                        r2 = r+delays[c]
                        if r2 < ntimes: tau_true[r,c] = time_im[r2]-time_im[r]
                
                tau_true = np.round(tau_true,6) #the time resolution is 1E-6 sec....
                
                
                print('\ncalculating the binned time delays...')
                #get a sorted array with all unique delays, excluding nan and inf
                a = np.sort(np.unique(tau_true))
                a = a[np.isfinite(a)]
                # "consolidate" list of delays, by grouping delays whose ratio is between 1
                # and rel_diff
                
                #define bins to which all delays will be assigned. binl[0..Nbins-1] and 
                #binh[0..Nbins-1] are the lower/upper bounds of the bins. 
                #We want the first bin to correspond to the first
                #delay only (usually 0 s or the smallest available lag):
                if a[0]==0:
                    epsilon = 1E-6
                else:
                    epsilon = 1E-6*a[0]
                binl = [a[0]-epsilon] #define the first bin so that it contains just a[0], to
                                        #within +/- epsilon
                binh = [a[0]+epsilon]
                hb = binh[0]  #the higher bound of the current bin
                for j in range(1,a.size):
                    if a[j] >= hb:
                        binl.append(a[j])
                        hb = rel_diff*a[j]
                        binh.append(hb)
                
                        
                #get indexes for consolidating data, consolidate tau_true, save relevant data
                print('\ncalculating the indexes for consolidating data.')
                print('This may take some time, for large cI files\n')
                print('***** NOTE: the message\n'+\
                      '\"RuntimeWarning: invalid value encountered in greater_equal...\"\n' + \
                      'is harmless\n')
                indexl,indexh = self.build_indexes(tau_true,binl,binh)
                tau_cons = self.consolidate(tau_true,indexl,indexh) 
                tau_mean = np.nanmean(tau_cons, axis = 0)        
                ##### save indexl, indexh, tau_cons, tau_mean (pickled python data)        
                outfile = folderin + 'consolidate_info.npz'
                np.savez(outfile,indexl=indexl,indexh=indexh,tau_cons=tau_cons,\
                         tau_mean=tau_mean,time_im=time_im)        
        
            self.tau = list(tau_mean)
            # "consolidate" cIs, i.e. for each t average them over delays tau that fall in
            # the same bin. Store in pandas dataframe and output to file consolidated cIs
            print('\nCalculating the consolidated cIs, be patient...')
            cI_cons = self.consolidate(np.asarray(cIraw.iloc[:,2:]),indexl,indexh) 
            
            #Store in pandas dataframe and output to file consolidated cIs
            #create a list with all delays in the format, e.g., 's1.58e-02' for a delay of 
            #1.58E-2 seconds. This list will be used as column names to output the 
            #consolidated cI file
            col_names = ['tsec','n','Iav']
            for t in tau_mean:
                col_names.append('s'+ format(t,'.3e'))
            cIcons = pd.DataFrame(index=range(ntimes),columns=col_names[0:3])
            cIcons['tsec'] = time_im
            cIcons['n'] = cIraw['n']
            cIcons['Iav'] = cIraw['Iav']
            cIcons2 = pd.DataFrame(cI_cons,index=range(ntimes),columns=col_names[3:])
            cIcons = pd.concat([cIcons,cIcons2],axis=1)
            
            print(ROInum)
            
            
            self.CI.append( cIcons)
            
        for i in range(self.nROI): 
            
            self.Iav.append(self.CI[i]['Iav'])
            self.CI[i].drop(['Iav'], axis=1,inplace=True)
            
            
            
            #save as text file
            fout = folderin + 'ROI' + str(ROInum).zfill(4) +'cI_ts.dat'  #ts stands for "all Times in Sec"
            cIcons.to_csv(fout,sep='\t',index=False,na_rep='nan') 
        
        return
    
    

    
    
    def Save_CSV(self):
        
        folder_CI_Processed = self.FolderName + '\\processed_CI\\'
        
        try:
            os.mkdir(folder_CI_Processed)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        for i in range(self.nROI):
            self.CI[i].to_csv(folder_CI_Processed + 'ROI' + str(i+1).zfill(4) + 'cI.dat',sep='\t',index=False,na_rep='NaN')
        
        
        
        tausave = pd.Series(self.tau)


        tausave.to_csv(folder_CI_Processed + 'lagtime.dat',sep='\t',index=False,na_rep='NaN')
        
           
        original = self.FolderName + '\\Input_101_CalcCI.dat'
        target = self.FolderName + '\\processed_CI\\Input_101_CalcCI.dat'

        shutil.copyfile(original, target)
        
        return
    
    def Quick_Load(self,FolderName):
        
        self.FolderName = FolderName
        self.LoadInput_101_CalCI()
        
 
        ROI_name_list=list()
            
        for i in range(self.nROI):
            ROI_name_list.append(str(1 + i).zfill(self.cI_file_digits))
            self.ROIfilelist.append(self.filename + ROI_name_list[i]+ self.extension) 
            self.FileList.append(self.FolderName + '\\' + self.filename + ROI_name_list[i] + self.extension)
                
        for i in range(self.nROI):
            self.CI.append(pd.read_csv(self.FileList[i], sep='\\t', engine='python'))
            
            
        a = pd.read_csv('E:\\Matteo\\PHD\\light_scattering\\20210622_silicaTM30_40ul_1Murea_100units_01Vf_300mW_15_dry13_SG\\Cam1\\exptime_0.070000\\out13\\processed_CI\\' + 'lagtime.dat', sep='\\t', engine='python')

        tauload = a.values.tolist()

        self.tau
        for i in range(len(tauload)):
            self.tau.append(tauload[i][0])
        

      
            
        for i in range(len(self.CI[0].columns)):
                if self.CI[0].columns[i].endswith(' s'):
                    
                    for char in self.CI[0].columns[i].split(' s'):
                        
                        if char.isdigit():
                            print('hola')
                            self.tau.append(float(char))

        self.Timepulse2 = True

        
        
        
        
        return
    
    def TimePulseLoad(self):
        
        filein = self.FolderName + '\\' + 'TimePulses.txt'
        cycle_data =  pd.read_csv(filein, sep="\t",header = None)
        n_pulses = int(cycle_data[0][0])
        pulse_time = np.asarray(cycle_data[0][1:n_pulses+1],dtype=np.float64)/1E6  #in sec
        cycle_dur = np.asarray(cycle_data[0][n_pulses+1],dtype=np.float64)/1E6  #in sec
        
        return pulse_time,n_pulses,cycle_dur
    
    def NoiseNormalization(self):
        
        for l in range(self.nROI):
            print('normalization of ROI '+str(l+1)+' over '+str(self.nROI))
            
            for i in range(len(self.CI[l].columns[3:])): 
                for j in range(self.CI[l].iloc[:,3+i].count()):
                    self.CI[l].iloc[j,3+i] = self.CI[l].iloc[j,3+i] / np.sqrt( self.CI[l].iloc[j,2] * self.CI[l].iloc[j+i+1,2] )  #attention the index i makes no sense!!!

        
        return
    
    def CIShow(self,which_ROI):
        
        folder_CI_graphs = self.FolderName + '\\CI_graphs'
        
        try:
            os.mkdir(folder_CI_graphs)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        
            
        
        if self.Timepulse2 == True: 
            time = self.CI[0]['tsec']
            plt.figure() 
            plt.title('CI ROI'+str(which_ROI).zfill(4))
            for i in range(len(self.CI[0].columns)-3):
                plt.plot(time,self.CI[which_ROI-1][self.CI[which_ROI-1].columns[i+3]],label=self.CI[which_ROI-1].columns[i+3],marker='.')
            plt.ylabel('CI ')
            plt.ylim([-0.1, 1.3])
            plt.xlabel('time [s]')
            plt.savefig(folder_CI_graphs+'\\CI_ROI'+str(which_ROI).zfill(4)+'.png', dpi=300)
            
            
        if self.Timepulse3 == True: 
            time = self.CI[0]['tsec']
            plt.figure() 
            plt.title('CI ROI'+str(which_ROI).zfill(4))
            for i in range(len(self.CI[0].columns)-3):
                plt.plot(time,self.CI[which_ROI-1][self.CI[which_ROI-1].columns[i+3]],label=self.CI[which_ROI-1].columns[i+3],marker='.')
            plt.ylabel('CI ')
            plt.ylim([-0.1, 1.3])
            plt.xlabel('time [s]')
            plt.savefig(folder_CI_graphs+'\\CI_ROI'+str(which_ROI).zfill(4)+'.png', dpi=300)
        
        
        elif self.Timepulse == True:

            time = self.CI[0]['tsec']   
            plt.figure() 
            plt.title('CI ROI'+str(which_ROI).zfill(4)) 
            for i in range(len(self.CI[0].columns)):
                if self.CI[0].columns[i].startswith('usec'):                  
                    plt.plot(time,self.CI[which_ROI-1][self.CI[which_ROI-1].columns[i]],label=self.CI[which_ROI-1].columns[i],marker='.')
                    plt.ylabel('CI ')
                    plt.ylim([-0.1, 1.3])
                    plt.xlabel('time [s]')
                    
            plt.savefig(folder_CI_graphs+'\\CI_ROI'+str(which_ROI).zfill(4)+'.png', dpi=300)
                    
            

                    
            for i in range(len(self.CI[0].columns)):
                if self.CI[0].columns[i].startswith('sec'):
                    plt.plot(time,self.CI[which_ROI-1][self.CI[which_ROI-1].columns[i]].tolist(),label=self.CI[which_ROI-1].columns[i],marker='.')
                    plt.ylabel('CI')
                    plt.ylim([-0.1, 1.3])
                    plt.xlabel('time [s]')
            
            plt.savefig(folder_CI_graphs+'\\CI_ROI'+str(which_ROI).zfill(4)+'.png', dpi=300)
            
            
        else :
            plt.figure() 
            for i in range(len(self.CI[0].columns)):
                if self.CI[0].columns[i].startswith('d'):
                    plt.plot(self.CI[which_ROI-1][self.CI[which_ROI-1].columns[i]].tolist(),marker='.')
                    plt.ylim([-0.1, 1.1])
            plt.show()        
            

        return
    
    def CIShowFancy(self,which_ROI):
        
        folder_CI_graphs = self.FolderName + '\\CI_graphs'
        
        try:
            os.mkdir(folder_CI_graphs)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
            
            
        if self.Timepulse3 == True: 
            
            n = len(self.CI[0].columns)-27
            colors = pl.cm.Reds(np.linspace(0,1,n))
            time = self.CI[0]['tsec']
            plt.figure() 
            plt.title('CI ROI'+str(which_ROI).zfill(4))
            
            for i in range(len(self.CI[0].columns)-27):
                if i % 1 == 0:
                    plt.plot(time,self.CI[which_ROI-1][self.CI[which_ROI-1].columns[i+3]],label=self.CI[which_ROI-1].columns[i+3],marker='.',color=colors[i],linestyle='')
            plt.ylabel('CI ')
            plt.ylim([-0.1, 1.3])
            plt.xlabel('time [s]')
            plt.savefig(folder_CI_graphs+'\\CI_ROI'+str(which_ROI).zfill(4)+'.png', dpi=300)
        
        
        return




    
    
    
class CIbead(CI):
    def __init__(self,n1,n2,wavelength,magnification):
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
        self.magnification = magnification
        
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
        self.h = []
        
        for i in range(len(self.ROI_x_pos)):
            inner_h,scattering_angle=sf.theta1_func(self.ROI_x_pos[i],Radius,self.indexrefbead,self.indexrefext)
            self.h.append(inner_h)
            self.scatt_angle.append(scattering_angle*360/(2*math.pi))
                
        return
    
    def SetqScatt(self,Radius):
        
        self.SetThetaScatt(Radius)
        
        q=4*np.pi*self.indexrefbead*np.sin(np.asarray(self.scatt_angle) / 2 * np.pi / 180 ) / (532*1e-9)
        
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
    
    
    def CIfindQdependence(self,decaytime,qvector):
        
        q = np.asarray(qvector)
        
        func = lambda m: q**m
        
        i = 0
        asymmetry = []
        exponent = [i]
        
        while i < 2:
            asymmetry.append(sf.AsymmetryCalculator(decaytime*func(i)))
            i = i + 0.1
            exponent.append(i)
            
        asy = np.asarray( asymmetry ) 
        
        index_asy_min = np.where( asy == np.min(asy))
        minimizing_exp = exponent[ index_asy_min[0][0] ]
        minimize_asymmetry = asymmetry[ index_asy_min[0][0] ]
         
        return minimizing_exp, minimize_asymmetry
    
    
    
    

    
#####################################################################   THIS CLASS SHOULD BE DELETED   #####################################################    
    
class CIdisplacements(CI):
    def __init__(self,n1,n2,wavelength,magnification):
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
        self.magnification = magnification
        self.cutcollectiondx = []
        self.cutcollectiondy = []
        self.Dx = []
        self.Dy = []
        
    def __str__(self):
        str_res  = '\n|---------------|'
        str_res += '\n| CIdisplacementsclass:    '
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
    
    def LoadConsolidateDisplacement(self, folderin,ROIlist,rel_diff,normalization=False): 
        
        self.FolderName = folderin
        self.Timepulse3 = True
        
        if folderin[-1] != '/' and folderin[-1] != '\\': folderin += '/'

        #Note: we assume that all the ROIs have been processed with the same set of 
        #parameters (same set of images, same set of time delays etc.)
        
        
        self.nROI = len(ROIlist)
        
        for ROInum in ROIlist:
            print('Processing data for ROI n.%d' % ROInum)
            
            #check if the npz file with the info for consolidating data exists
            in_npzfile = folderin + 'consolidate_info.npz'
            calc_cons = True
            try:
                npzfile = np.load(in_npzfile)
                indexl = npzfile['indexl']
                indexh = npzfile['indexh']
                tau_cons = npzfile['tau_cons']
                tau_mean = npzfile['tau_mean']
                time_im = npzfile['time_im']
            except:
                calc_cons = True
        
            
            filein = folderin + 'ROI' + str(ROInum).zfill(4) + 'Disp.dat'
            cIraw = pd.read_csv(filein, sep="\t") #read cI file (note: fills with NaN missing data)
            
          
                
            
            ntimes = cIraw['n'].size
            if calc_cons:
                #get list of delays (in number of images):
                delays = np.asarray(cIraw.columns[2:],dtype = str)
                for i in range(delays.size): delays[i] = delays[i].replace('dx','')  
                for i in range(delays.size): delays[i] = delays[i].replace('dy','') 
                delays = delays.astype(int)
                ndelays = delays.size
            
                #get list of time delays within a cycle
                #TO DO: find time of each image in different ways (TimePulses, Guillaume's
                #file with image time, ImagesLog etc.) 
                filein = folderin + 'TimePulses.txt'
                cycle_data =  pd.read_csv(filein, sep="\t",header = None)
                n_pulses = int(cycle_data[0][0])
                pulse_time = np.asarray(cycle_data[0][1:n_pulses+1],dtype=np.float64)/1E6  #in sec
                cycle_dur = np.asarray(cycle_data[0][n_pulses+1],dtype=np.float64)/1E6  #in sec
                
                #time at which each image was taken (in sec, t=0 at the beginning of the cI file)
                
                time_im = np.zeros(ntimes,dtype = np.float64) 
                time_im[0] =  pulse_time[0]
                for j in range(0,time_im.size):
                   time_im[j] = cycle_dur*(j//n_pulses) + pulse_time[j%n_pulses]
                
                #time delay between all pairs of images for which cI has been calculated
                print('\ncalculating the time delays between all pairs of images...')
                tau_true = np.ones((ntimes,ndelays),dtype = np.float64)*np.nan
                for r in range(ntimes):
                    for c in range(ndelays):
                        r2 = r+delays[c]
                        if r2 < ntimes: tau_true[r,c] = time_im[r2]-time_im[r]
                
                tau_true = np.round(tau_true,6) #the time resolution is 1E-6 sec....
                
                
                print('\ncalculating the binned time delays...')
                #get a sorted array with all unique delays, excluding nan and inf
                a = np.sort(np.unique(tau_true))
                a = a[np.isfinite(a)]
                # "consolidate" list of delays, by grouping delays whose ratio is between 1
                # and rel_diff
                
                #define bins to which all delays will be assigned. binl[0..Nbins-1] and 
                #binh[0..Nbins-1] are the lower/upper bounds of the bins. 
                #We want the first bin to correspond to the first
                #delay only (usually 0 s or the smallest available lag):
                if a[0]==0:
                    epsilon = 1E-6
                else:
                    epsilon = 1E-6*a[0]
                binl = [a[0]-epsilon] #define the first bin so that it contains just a[0], to
                                        #within +/- epsilon
                binh = [a[0]+epsilon]
                hb = binh[0]  #the higher bound of the current bin
                for j in range(1,a.size):
                    if a[j] >= hb:
                        binl.append(a[j])
                        hb = rel_diff*a[j]
                        binh.append(hb)
                
                        
                #get indexes for consolidating data, consolidate tau_true, save relevant data
                print('\ncalculating the indexes for consolidating data.')
                print('This may take some time, for large cI files\n')
                print('***** NOTE: the message\n'+\
                      '\"RuntimeWarning: invalid value encountered in greater_equal...\"\n' + \
                      'is harmless\n')
                indexl,indexh = self.build_indexes(tau_true,binl,binh)
                tau_cons = self.consolidate(tau_true,indexl,indexh) 
                tau_mean = np.nanmean(tau_cons, axis = 0)        
                ##### save indexl, indexh, tau_cons, tau_mean (pickled python data)        
                outfile = folderin + 'consolidate_info.npz'
                np.savez(outfile,indexl=indexl,indexh=indexh,tau_cons=tau_cons,\
                         tau_mean=tau_mean,time_im=time_im)        
        
            self.tau = list(tau_mean)
            # "consolidate" cIs, i.e. for each t average them over delays tau that fall in
            # the same bin. Store in pandas dataframe and output to file consolidated cIs
            print('\nCalculating the consolidated cIs, be patient...')
            cI_cons = self.consolidate(np.asarray(cIraw.iloc[:,2:]),indexl,indexh) 
            
            #Store in pandas dataframe and output to file consolidated cIs
            #create a list with all delays in the format, e.g., 's1.58e-02' for a delay of 
            #1.58E-2 seconds. This list will be used as column names to output the 
            #consolidated cI file
            col_names = ['tsec','n','Iav']
            for t in tau_mean:
                col_names.append('s'+ format(t,'.3e'))
            cIcons = pd.DataFrame(index=range(ntimes),columns=col_names[0:3])
            cIcons['tsec'] = time_im
            cIcons['n'] = cIraw['n']
            #cIcons['Iav'] = cIraw['Iav']
            cIcons2 = pd.DataFrame(cI_cons,index=range(ntimes),columns=col_names[3:])
            cIcons = pd.concat([cIcons,cIcons2],axis=1)
            
            print(ROInum)
            
            
            self.CI.append( cIcons)
            
        for i in range(self.nROI): 
            
            #self.Iav.append(self.CI[i]['Iav'])
            self.CI[i].drop(['Iav'], axis=1,inplace=True)
            
            
            
            #save as text file
            fout = folderin + 'ROI' + str(ROInum).zfill(4) +'cI_ts.dat'  #ts stands for "all Times in Sec"
            cIcons.to_csv(fout,sep='\t',index=False,na_rep='nan') 
            
        # start the cycle here
        for i in range(self.nROI):
            dx = pd.DataFrame(self.CI[0]['tsec'])
            dy = pd.DataFrame(self.CI[0]['tsec'])
            
            for j in range(len(self.CI[0].columns)-3):
                    if j % 2 == 0:
                        dx.insert(len(dx.columns),str(len(dx.columns)),self.CI[i][self.CI[i].columns[j+2]])
                        
                    if j % 2 != 0:
                        dy.insert(len(dy.columns),str(len(dy.columns)),self.CI[i][self.CI[i].columns[j+2]])
                        
            self.Dx.append(dx)
            self.Dy.append(dy)
            
            
        return

    def CIShowDisplacement(self,which_ROI):
        
        folder_CI_graphs = self.FolderName + '\\CI_graphs'
        
        try:
            os.mkdir(folder_CI_graphs)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
                
        if self.Timepulse3 == True: 
            time = self.CI[0]['tsec']
            plt.figure() 
            plt.title('ROI'+str(which_ROI).zfill(4))
            for i in range(len(self.Dx[0].columns)-3):
            
                plt.plot(time,self.Dx[which_ROI-1][self.Dx[which_ROI-1].columns[i+3]],label=self.Dx[which_ROI-1].columns[i+3],marker='.')
            plt.ylabel('dx [px]')
            #plt.ylim([-10, 1.3])
            plt.xlabel('time [s]')
            plt.savefig(folder_CI_graphs+'\\CI_ROI'+str(which_ROI).zfill(4)+'.png', dpi=300)
                
            plt.figure() 
            plt.title('ROI'+str(which_ROI).zfill(4))
            for i in range(len(self.Dy[0].columns)-3):
            
                plt.plot(time,self.Dy[which_ROI-1][self.Dy[which_ROI-1].columns[i+3]],label=self.Dy[which_ROI-1].columns[i+3],marker='.')
            plt.ylabel('dy [px]')
            #plt.ylim([-0.5, 1.3])
            plt.xlabel('time [s]')
            plt.savefig(folder_CI_graphs+'\\CI_ROI'+str(which_ROI).zfill(4)+'.png', dpi=300)

    def CIRemovedelay(self,ndelay):
        
        for i in range(self.nROI):
            for j in range(ndelay):
                #self.CI[i][self.CI[i].columns[-3]].drop(columns=[self.CI[i].columns[-3]])
                self.CI[i].drop(columns=[self.CI[i].columns[-1]], axis = 1, inplace = True)
                       
        return         
    
    def CISelectTime(self,time):
        
        for i in range(self.nROI):
            cx = np.asarray(self.Dx[i].iloc[[round(time)]])
            cy = np.asarray(self.Dy[i].iloc[[round(time)]])
            self.cutcollectiondx.append(np.delete(cx,0))
            self.cutcollectiondy.append(np.delete(cy,0))
        
        
        return
        
        
        
    
        
      
        
