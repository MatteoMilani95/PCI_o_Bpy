# -*- coding: utf-8 -*-
"""
Created on Thu May 12 17:54:16 2022

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
        self.CI_list = []
        self.nROI = 0
        
        self.Dispx_list = []
        self.Dispy_list = []
        
        self.folderin = []
        self.folderout = []
        
        self.tau_seconds = np.asarray([0])
        
        
        #self.Input_101 =[]
    def __repr__(self): 
        return '<ROI: fn%s>' % (self.FolderName)
    
    def __str__(self):
        str_res  = '\n|---------------|'
        str_res += '\n| CIts class: '
        str_res += '\n|--------------------+--------------------|'
        str_res += '\n| objects: '
        str_res += '\n|--------------------+--------------------|'
        str_res += '\n| folderin        : ' + str(self.folderin)
        str_res += '\n| folderout       : ' + str(self.folderout)
        str_res += '\n| CI_list         : ' 
        str_res += '\n| tau_seconds     : ' + str(len(self.tau_seconds))
        str_res += '\n| nROI            : ' + str(self.nROI)
        str_res += '\n| Dispx_list      : ' 
        str_res += '\n| Dispy_list      : ' 
        str_res += '\n|--------------------+--------------------|'
        str_res += '\n| methods: '
        str_res += '\n|--------------------+--------------------|'
        str_res += '\n| prepare_folder : creates a folder if not existing'
        str_res += '\n| get_delays     : get delay list from CI'
        str_res += '\n| cI_to_cI_ts    : load and transform in CI_ts'
        str_res += '\n|--------------------+--------------------|'
        return str_res
    
    
    
    
    
    
    #########################################
    ### START functions to process cI files
    
    
        
    
    
    def d0_normalize(self,cI_df,delays):
        """
        Normalizes the cIs using the following relationship:
        c_I(t,tau) --> c_I(t,tau) / [c_I(t,tau=0)*c_I(t+tau, tau=0)]^0.5
        IMPORTANT: before doing all the maths, we convert the dataframe to a 
        numpy array, because for large sets of data numpy is much, much faster!
    
        Parameters
        ----------
        cI_df : Pandas DataFrame
                  dataframe with cI data, as provided by Analysis 101
        
        delays : numpy 1D array, int
                delays for each column of cI data, as obtained using 
                get_delays(fname)
    
        Returns
        -------
        norm_cI_df : Pandas DataFrame
                 dataframe with normalised cI data (in the same format as that 
                 provided by Analysis in case further data processing is needed)
    
        """    
      
        #make sure that the number of columns in the dataframe is consistent with
        #the number of delays in 'delays'
        if len(cI_df.columns) != delays.shape[0]+2: #delays does not count the n and Iave columns...
            raise NameError('d0_normalize(): # of dataframe columns '+\
                'inconsistent with # of delays')
                
        cI = cI_df.to_numpy(dtype = float,copy=True)
        cI_norm = np.ones(cI.shape)*np.nan
        cId0 = cI[:,2]
        
        #Normalize cI values (work on numpy array to speed up)
        for c in range(2,cI.shape[1]):
            cId0rolled = np.roll(cId0,-delays[c-2])
            normfact = cId0*cId0rolled
            #assign nan if normfact too small or negative (which should not happen)
            problem = np.where(normfact < 1E-12)
            normfact[problem] = np.nan
            cI_norm[:,c] = cI[:,c]/np.sqrt(normfact)
    
                    
        #copy first two columns of cI to cI_norm:
        np.copyto(cI_norm[:,:2],cI[:,:2])
        
        norm_cI_df = pd.DataFrame(data=cI_norm,index=None,columns=cI_df.columns,dtype=float)
        norm_cI_df = norm_cI_df.astype({'n': 'int32'})  #set back first col (col 'n') to int type
        return norm_cI_df
    
    
    
    def d0_normalize_many_ROIs(self,folderin,ROIlist,verbose=True,ROIdigits=4):
        """
        Normalizes the raw cI file to reduce noise due to d0 fluctuations, using
        d0_normalize. All raw cI files corresponing to ROIs in ROIlist are 
        processed. The corresponding '..._norm.dat' fils are written in folderin
    
    
        Parameters
        ----------
        folderin : TYPE str
            DESCRIPTION: input folder where the cI files to be processed are
        ROIlist : TYPE list of int
            DESCRIPTION: list of the ROIs to process, e.g. [1,3,4]
        verbose : TYPE bool, optional, the default is True
            DESCRIPTION: True/False to/not to printy various status messages.
        ROIdigits : TYPE int, optional, the default is 4.
            DESCRIPTION: # of digits in thz cI files. E.g. 4 in ROI0023cI.dat
    
        Returns
        -------
        None.
    
        """
        folderin = sf.prepare_folder(folderin)
        #Note: we assume that all the ROIs have been processed with the same set of 
        #parameters (same set of images, same set of time delays etc.)
        
        files_to_do = []
        files_out = []
        filetypes = ['cI','cIcr']  #types of files to be normalized
        
        for ROInum in ROIlist:
            for ft in filetypes:   
                #filename radix:
                radix = folderin + 'ROI' + str(ROInum).zfill(ROIdigits) + ft
                doprocess = True
                #check if processed file already exists
                filein = radix + '_norm.dat'
                if os.path.isfile(filein):
                    doprocess = False
                    if verbose: print('%s\nalready exists, skipping it' % filein)
                #check if input file actually exists
                filein = radix + '.dat'
                if not os.path.isfile(filein):
                    doprocess = False
                    if verbose: print('%s\nnot found, skipping it' % filein)
        
                if doprocess:        
                    files_to_do.append(filein) 
                    files_out.append(radix + '_norm.dat')
    
    
               
        if len(files_to_do) < 1: return
        
        for i, fin in enumerate(files_to_do):
            if verbose: print('Processing\n%s' % fin)
            delays = sf.get_delays(fin)
            cI_df = pd.read_csv(fin, sep="\t")
            norm_cI_df = self.d0_normalize(cI_df,delays)
            norm_cI_df.to_csv(files_out[i],sep='\t',index=False,na_rep='nan',float_format="%.6f")
                                 
        
        return
        
    
    def consolidate(self,in2darray,il,ih):
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
    
    
    
    def build_indexes(self,tau_true,binl,binh):
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
    
    
    
    def cI_to_cI_ts(self,folderin,ROIlist=[1],reldiff=1.05,folderout=None,verbose=True,\
                    ROIdigits=4):
        """
        Converts the cI files generated by Analysis.exe (C/C++ soft by Luca) into
        the cI_ts format, where all times and time delays are in sec, rather than 
        in number of images. Note: will skip input files not found or for which
        the corresponding '_ts' file already exists
    
        Parameters
        ----------
        folderin : TYPE str
            DESCRIPTION: input folder where the cI files to be processerd are
        ROIlist : TYPE list of int, default is [1]
            DESCRIPTION: list of the ROIs to process, e.g. [1,3,4]
        reldiff : TYPE float, default is 1.05
            DESCRIPTION: this script will "consolidate" cI data by averaging over 
                         sets of delays whose ratio is between 1 and reldiff. 
                         E.g.: if reldiff = 1.05, delays of 10 s and 11 s will be 
                         treated as distinct, while cI data for delays of 10 s and 
                         10.4 s will be averaged together
        folderout : TYPE str or None, default is None (will set folderout=folderin)
            DESCRIPTION: output folder where the processed cI files will be written
                         if None, folderout = folderin. folderout will be created,
                         if needed
        verbose : TYPE bool, optional, the default is True
            DESCRIPTION: True/False to/not to printy various status messages.
                         .
        ROIdigits : TYPE int, optional, the default is 4.
            DESCRIPTION: # of digits in thz cI files. E.g. 4 in ROI0023cI.dat
    
        Returns
        -------
        None.
    
        """
        
        self.nROI = len(ROIlist)
        
        self.folderin = sf.prepare_folder(folderin)
        if folderout == None: self.folderout = self.folderin
        else: self.folderout = sf.prepare_folder(folderout)
        
        
        
        #Note: we assume that all the ROIs have been processed with the same set of 
        #parameters (same set of images, same set of time delays etc.)
        
    #####build a list of cI-like files to be processed, i.e. excluding the 
        #displacement files. A file is to be processed if:
        # 1) the 'regular' version of that file exists
        # 2) the file has not been processed yet (i.e. the corresponding '_ts'
        #    version does not exist)
        
        cI_to_process = []
        cI_to_output = []
        to_consolidate = ['cI','cIcr','cIcr_norm','cI_norm']
        for suffix in to_consolidate:
            for ROInum in ROIlist:
                doskip = False
                fin = self.folderin+'ROI'+str(ROInum).zfill(ROIdigits)+suffix+'.dat'
                fout = self.folderout+'ROI'+str(ROInum).zfill(ROIdigits)+suffix+'_ts.dat'
                if not os.path.isfile(fin):
                    if verbose: 
                        print('\ncI_to_cI_ts():\n%s\n not found, skipping it' %fin)
                    doskip = True
                if os.path.isfile(fout):
                    if verbose: 
                        print('\ncI_to_cI_ts():\n%s\n already exists, skipping it'\
                               %fout)
                        
                    doskip = True
                if not doskip:
                    cI_to_process.append(fin)
                    cI_to_output.append(fout)
    
               
        if len(cI_to_process) < 1: 
            print('\ncI_to_cI_ts(): no cI files to process, doing nothing')
            return
    
        #read the first cI file to be processed in order to get useful info
        filein = cI_to_process[0]
        print(cI_to_process)
        cIraw = pd.read_csv(filein, sep="\t") #read cI file (note: fills with NaN missing data)
        ntimes = cIraw['n'].size
                    
                    
    #####build a list of displacement files to be processed 
        #A file is to be processed if:
        # 1) the 'regular' version of that file exists
        # 2) the file has not been processed yet (i.e. the corresponding '_ts'
        #    version does not exist)
        
        disp_to_process = []
        dispx_to_output = []
        dispy_to_output = []
        suffix = 'Disp'
        for ROInum in ROIlist:
            doskip = False
            fin = self.folderin+'ROI'+str(ROInum).zfill(ROIdigits)+suffix+'.dat'
            foutx = self.folderout+'ROI'+str(ROInum).zfill(ROIdigits)+suffix+'_x_ts.dat'
            fouty = self.folderout+'ROI'+str(ROInum).zfill(ROIdigits)+suffix+'_y_ts.dat'
            if not os.path.isfile(fin):
                if verbose: 
                    print('\ncI_to_cI_ts():\n%s\n not found, skipping it' %fin)
                doskip = True
            if os.path.isfile(foutx) or os.path.isfile(fouty):
                if verbose: 
                    print('\ncI_to_cI_ts():\n%s or\n%s\n already exists, skipping both'\
                           % (foutx,fouty) )
                doskip = True
            if not doskip:
                disp_to_process.append(fin)
                dispx_to_output.append(foutx)
                dispy_to_output.append(fouty)
           
    
    #####check if the npz file with the info for consolidating data exists, create
        #it if needed
        in_npzfile = self.folderout + 'consolidate_info.npz'
        calc_cons = False
        try:
            npzfile = np.load(in_npzfile)
            indexl = npzfile['indexl']
            indexh = npzfile['indexh']
            tau_cons = npzfile['tau_cons']
            tau_mean = npzfile['tau_mean']
            self.tau_seconds = tau_mean
            time_im = npzfile['time_im']
        except:
            calc_cons = True
        print(calc_cons)
        #if the npz file does not exist, create it
        if calc_cons:
            #read from report_101.txt the first /last processed dataset & image
            yesreport = False
            try:
                f = open(self.folderin +'report_101.txt')
            except IOError:
                print('\nreport_101.txt not found: assuming that all MI files were used to calculate cIs')
            else:
                yesreport = True
                with f:
                    reportstr = f.read()
                    ii = reportstr.rfind('First processed dataset:')
                    if ii==-1:
                        print('\nInfo on first/last processed images not ' +\
                              'found in report_101.txt:\nassuming that ' +\
                              'all MI files were used to calculate cIs')
                        yesreport = False
        
            if yesreport:
                numlist = re.findall(r'\d+',reportstr[ii:] ) #list of numbers
                #get first dataset & image that were processed
                firstdataset = int(numlist[0])
                firstimage = int(numlist[1])
                #get last dataset & image that were processed
                lastdataset = int(numlist[2])
                lastimage = int(numlist[3])
                if verbose: print('\nreport_101.txt: the cI file(s) correspond to images\n'+\
                    'from dataset, image: %d, %d\nto   dataset, image: %d, %d\n'\
                    %(firstdataset,firstimage,lastdataset,lastimage))
            
            
            #get list of delays (in number of images):
            delays = sf.get_delays(filein)
            print('ciao')
            print(delays)
            ndelays = delays.size        
            
            #time at which each image was taken (in sec, t=0 at the beginning of the cI file)
            filein = self.folderin + 'ImagesTime.dat'
            try:
                images_time =  pd.read_csv(filein, sep="\t")
            except IOError: 
                raise NameError('cI_to_cI_ts(): ImagesTime.dat not found, unable\n',
                                'to retrieve acquisition time of all images')
            #get time units
            colnames = images_time.columns.values.tolist()
            ii = colnames[2].find('.')
            tunitstr = colnames[2][ii+1:]
            if tunitstr == 'msec': tunitstr = '1E-3s' #for backward compatibility
            tunitstr = tunitstr[:-1]
            tconv = float(tunitstr) #conversion factor --> all times in s
            
            #slice dataset to use only times corresponding to images
            #processed in the cI file(s)
            if yesreport:
                firstindex = images_time.index[(images_time['Dataset'] == firstdataset)\
                        & (images_time['nImages'] == firstimage)][0]
                lastindex = images_time.index[(images_time['Dataset'] == lastdataset)\
                        & (images_time['nImages'] == lastimage)][0]
                images_time = images_time[firstindex:lastindex+1]
            
            
            
            #get time of each image, in sec
            time_im = (images_time[colnames[3]]).to_numpy(dtype = np.float64)
            time_im *= tconv  #convert to sec
           
            #time delay between all pairs of images for which cI has been calculated
            if verbose: 
                print('\nConsolidating\n%s...' % filein)
                print('\ncalculating the time delays between all pairs of images...')
            tau_true = np.ones((ntimes,ndelays),dtype = np.float64)*np.nan
            for r in range(ntimes):
                for c in range(ndelays):
                    r2 = r+delays[c]
                    if r2 < ntimes: tau_true[r,c] = time_im[r2]-time_im[r]
            
            tau_true = np.round(tau_true,6) #the time resolution is 1E-6 sec....
            
            
            if verbose: print('\ncalculating the binned time delays...')
            #get a sorted array with all unique delays, excluding nan and inf
            a = np.sort(np.unique(tau_true))
            a = a[np.isfinite(a)]
            # "consolidate" list of delays, by grouping delays whose ratio is between 1
            # and reldiff
            
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
                    hb = reldiff*a[j]
                    binh.append(hb)
            
                    
            #get indexes for consolidating data, consolidate tau_true, save relevant data
            if verbose: 
                print('\ncalculating the indexes for consolidating data.')
                print('This may take some time, for large cI files\n')
                print('***** NOTE: the message\n'+\
                  '\"RuntimeWarning: invalid value encountered in greater_equal...\"\n' + \
                  'is harmless\n')
            indexl,indexh = self.build_indexes(tau_true,binl,binh)
            tau_cons = self.consolidate(tau_true,indexl,indexh) 
            tau_mean = np.nanmean(tau_cons, axis = 0)
            self.tau_seconds = tau_mean
            print('hole')
            ##### save indexl, indexh, tau_cons, tau_mean (pickled python data)        
            outfile = self.folderout + 'consolidate_info.npz'
            np.savez(outfile,indexl=indexl,indexh=indexh,tau_cons=tau_cons,\
                     tau_mean=tau_mean,time_im=time_im)
            
    
    
    #####Consolidate all cI-like files
        for i, filein in enumerate(cI_to_process):
            cIraw = pd.read_csv(filein, sep="\t") #read cI file (note: fills with NaN missing data)
            if verbose: print('\nConsolidating\n%s...' % filein)
            cI_cons = self.consolidate(np.asarray(cIraw.iloc[:,2:]),indexl,indexh) 
            
            #Store in pandas dataframe and output to file consolidated cIs
            #create a list with all delays in the format, e.g., 's1.58e-02' for a delay of 
            #1.58E-2 seconds. This list will be used as column names to output the 
            #consolidated cI file
            col_names = ['tsec','n','Iav']
            for t in tau_mean:
                col_names.append('s'+ "{:.3e}".format(t))
            cIcons = pd.DataFrame(index=range(ntimes),columns=col_names[0:3])
            cIcons['tsec'] = time_im
            cIcons['n'] = cIraw['n']
            cIcons['Iav'] = cIraw['Iav']
            cIcons2 = pd.DataFrame(cI_cons,index=range(ntimes),columns=col_names[3:])
            cIcons = pd.concat([cIcons,cIcons2],axis=1)
            self.CI_list.append(cIcons)
            #save as text file
            fout = cI_to_output[i]
            cIcons.to_csv(fout,sep='\t',index=False,na_rep='nan',float_format="%.6f") 
    
    
    #####Consolidate all displacement files
        #note that we need to treat separately the case of ROIxxxxDisp.dat, because  
        #this file has a slightly different structure. We will output two separate
        # _ts displacement files, for x and y displacements, respectively    
        for i, filein in enumerate(disp_to_process):
            if verbose: print('\nConsolidating\n%s...' % filein)
            disp = pd.read_csv(filein, sep="\t",nrows=1) #just to get number of cols
            ncols = disp.shape[1]
            #columns to be used. Note that for consistency with cI file format
            #we also include column #0 (i.e. the image number column)
            dx_cols = [0] + [i for i in range(1,ncols,2)]
            dy_cols = [0] + [i for i in range(2,ncols+1,2)]
            disp_x = pd.read_csv(filein, sep="\t",usecols = dx_cols)
            disp_y = pd.read_csv(filein, sep="\t",usecols = dy_cols)
            #consolidate displacements along x, save them
            disp_consx = self.consolidate(np.asarray(disp_x.iloc[:,1:]),indexl,indexh) 
            dispconsx = pd.DataFrame(index=range(ntimes),columns=col_names[0:2])
            dispconsx['tsec'] = time_im
            dispconsx['n'] = cIraw['n']
            dispconsx2 = pd.DataFrame(disp_consx,index=range(ntimes),columns=col_names[3:])
            dispconsx = pd.concat([dispconsx,dispconsx2],axis=1)
            #save consolidated x displacement as text file
            fout = dispx_to_output[i]
            dispconsx.to_csv(fout,sep='\t',index=False,na_rep='nan',float_format="%.6f") 
    
            #now consolidate displacements along y, save them
            disp_consy = self.consolidate(np.asarray(disp_y.iloc[:,1:]),indexl,indexh) 
            dispconsy = pd.DataFrame(index=range(ntimes),columns=col_names[0:2])
            dispconsy['tsec'] = time_im
            dispconsy['n'] = cIraw['n']
            dispconsy2 = pd.DataFrame(disp_consy,index=range(ntimes),columns=col_names[3:])
            dispconsy = pd.concat([dispconsy,dispconsy2],axis=1)
            fout = dispy_to_output[i]
            dispconsy.to_csv(fout,sep='\t',index=False,na_rep='nan',float_format="%.6f") 
    
    
    ######all done!
        print('\ncI_to_cI_ts(): done!')
        return
    
    
    
    def plot_cI(self,folderin,ROIlist,suffix=[],savename=[],ROIdigits=4,axes=[]):
        """
        Plots and saves a figure of the cIs. One plot per ROI and one additional
        plot with the intensity vs. time for all ROIs (as per 29/3/2022)
    
    
        Parameters
        ----------
        folderin : TYPE str
            DESCRIPTION: input folder where the cI files to be plotted are
        ROIlist : TYPE list of int
            DESCRIPTION: list of the ROIs whose cIs are to be plotted, e.g. [1,3,4]
        suffix : TYPE list of str
            DESCRIPTION: list of suffixes to designate the kind of cI files to be 
                        plotted (e.g. 'cI_norm_ts.dat' to plot data for 
                        ROIxxxxcI_norm_ts.dat). If list is empty, sets each element 
                        to 'cI_ts.dat'. If suffix contains less elements than,
                        ROIlist, the last element will be duplicated up to filling
                        a list of same length as ROIlist
        savename : TYPE list of str, the default is []
            DESCRIPTION: list of filenames for saving the plots. If empty, uses
                         default names (e.g. ROI0001cI_ts.pdf etc.). 
                         savename=[None] not to save the plot
        ROIdigits : TYPE int, optional, the default is 4.
            DESCRIPTION: # of digits in the cI files. E.g. 4 in ROI0023cI_ts.dat
        axes : TYPE list of matplotlib axis instances.
            DESCRIPTION: a list of existing instances of matplotlib axes where to  
            plot the data. If the list length is different from the number of ROIs
            in ROIlist+1, or the list is empty, new figures and axes will be 
            created (one per ROI + one with intensity vs time).
        
    
        Returns
        -------
        figlist,axislist : list of the figure and axis instances obtained on generating 
                    the figure
        
    
        """
        if folderin[-1] != '\\' and folderin[-1] != '/': folderin += '/'
        figlist = []
    
        if len(axes) != len(ROIlist)+1:  #create new figures,1 per ROI + 1 for I(t)
            axislist = []
            newfig = True
        else:
            newfig = False
            axislist = axes
            
        if len(savename) == 1 and savename[0] == None: dosave = False
        else: dosave = True
        if dosave and len(ROIlist) != len(savename) and newfig == True: 
            autoname=True
        else: autoname = False
        
        ls = len(suffix)
        if ls == 0:
            suffix = ['cI_ts.dat']*len(ROIlist)
        else:
            dl = len(ROIlist)-ls          
            if ls > 0: suffix += [suffix[-1] for i in range(dl)]
        
        
        #prepare fig, axes for I(t) plot
        if newfig:
            fig_I,axI = plt.subplots(1,figsize=(10,10))  
        else:
            axI = axes[-1]
            fig_I = axI.get_figure()
            
            
        #plot the cI data for each ROI
        for ii, ROInum in enumerate(ROIlist):
            last = ii==len(ROIlist)-1  #last = True when procssing the last ROI
            #filename with cI data (should be a _ts file)
            filein = 'ROI' + str(ROInum).zfill(ROIdigits) + suffix[ii]  
            delay_sec = sf.get_delays(folderin + filein)
            cIcons = pd.read_csv(folderin + filein, sep="\t") #read cI file (note: fills with NaN missing data)
            #plot consolidated cIs, skipping the nans
            if newfig:
                fig_cI,axcI = plt.subplots(1,figsize=(10,10))  
                figlist.append(fig_cI)
                axislist.append(axcI)
            else:
                axcI = axes[ii]
                fig_cI = axcI.get_figure()
            for j in range(3,cIcons.shape[1]):
                if newfig or last:
                    lbl = '$\\tau=$%.2e s' %delay_sec[j-3]
                else:
                    lbl=''
                good = np.isfinite(np.asarray(cIcons.iloc[:,j],dtype = np.float))
                axcI.plot(cIcons['tsec'][good],cIcons.iloc[:,j][good],\
                          label = lbl)
            fig_cI.suptitle(folderin)
            axcI.set_title(filein)
            axcI.set_xlabel('$t$ (s)')
            axcI.set_ylabel('$c_I(t, \\tau)$')
            if newfig:
                fig_cI.legend()
                fig_cI.show()
            else:
                if ii==len(ROIlist)-1:  #one single legend (all cIs are supposed to
                    #correspond to the same list of delays)
                    fig_cI.legend()
                    fig_cI.show()
    
            #add I(t) for current ROI in the last plot
            axI.plot(cIcons['tsec'][good],cIcons.iloc[:,2][good],\
                      label = 'ROI' + str(ROInum).zfill(ROIdigits)+suffix[ii])
            
            #save cI figure
            
            #TODO: problems if we want to save une single figure (e.g. if axes != [])
            # In this case, we should save only at the very end, after plottin I(t)
            
            if dosave and newfig:
                if autoname:
                    figfile = (folderin +filein)[:-3] +\
                                'pdf'
                else:
                    if savename[ii][-4:].lower() != '.pdf': savename[ii]+='.pdf'
                    figfile = folderin + savename[ii]
        
                fig_cI.suptitle('This file: '+figfile +'\nData from '+ folderin)
                fig_cI.savefig(figfile)
                os.startfile(figfile)   # open the figure in the default pdf reader 
    
    
    
        #add fig, axes of the I(t) plot, if needed
        if newfig:
            figlist.append(fig_I)
            axislist.append(axI)
    
    
        fig_cI.suptitle(folderin)
        axcI.set_title(filein)
        axI.set_xlabel('$t$ (s)')
        axI.set_ylabel('$I(t)$')
        axI.legend()
        fig_I.show()
    
    
        #save cI figure
        if dosave:
            if autoname:
                if newfig: #new figures were created, separate figs for cI and I(t)
                    figfile = folderin + suffix[0][:-4] + '_I(t).pdf'
                else: #one single figure with all plots (cI and I(t))
                    figfile = folderin + suffix[0][:-4] + '_all.pdf'
            else:
                if savename[-1][-4:].lower() != '.pdf': savename[-1]+='.pdf'
                figfile = folderin + savename[-1]
    
            if newfig:
                fig_I.suptitle('This file: '+figfile)
            axI.set_title('Data from '+ folderin)
            fig_I.savefig(figfile)
            os.startfile(figfile)   # open the figure in the default pdf reader 
    
    
        
        return figlist,axislist