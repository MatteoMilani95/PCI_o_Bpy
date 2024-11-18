# -*- coding: utf-8 -*-
"""
Created on Thu May 12 18:06:41 2022

@author: Matteo
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import pickle as pk
from typing import Union, List, Tuple
import copy
from PCI_o_B import SharedFunctions as sf
from PCI_o_B import CI_ts_file as CI
from scipy.optimize import curve_fit



class Corr_func:
    """
    A simple class to create, load, and manipulate g2-1 correlation functions
    obtained from cI files in the _ts format (all times expressed in s)
    """


    
    def __init__(self,age_info: Union[int,List[Tuple[float]]]=1,\
                      roi_info: Union[int,List[int]]=1,\
                      folderin: Union[str,None]=None,\
                      suffix:   Union[str,None]=None,\
                      ROIdigits:int = 4):
        """
        class initiator. Noramlly called with info where to read cI file
        TODO complete parameter description
            
        Parameters
        ----------
        age_info: Union[int,List[tuple[float]]], default is 1
        DESCRIPTION. if int: number of age intervals to be included in the list 
                     (each element will be set to (0,1E6), which in practice
                     means average over the whole set of cI data)
                     if list of tuples: each tuple is like (tinit,tend),
                     tinit and tend in s designate starting/ending time of the 
                     intervals over which cI data are to be averaged
                     
        roi_info : Union[int,List[int]], default is 1
        DESCRIPTION. if int: number of ROIs to be included in the list 
                     (each element will be set to 0)
                     if list of int: list of ROI id numbers
                     
        
                     
                     
        """
        self.__age = sf.to_age_list(age_info)
        self.__n_age = len(self.__age)
        self.__roi = sf.to_ROI_list(roi_info) #get ROI list
        self.__n_roi = len(self.__roi)
        #file names from which the data used to build g2-1 are taken
        self.__sourcefile = ['']*self.__n_roi
        self.__n_tau = 1
        #array with delays (in s). Will be updated when loading from cI file
        self.__tau = np.zeros(self.__n_tau,dtype=np.float) 
        #numpy 3D array to store g2-1:
        self.__g2 = np.zeros(shape=(self.__n_age,self.__n_roi,self.__n_tau),\
                           dtype=np.float)
        self.__decaytime = np.zeros(shape=(self.__n_age,self.__n_roi,self.__n_tau),\
                           dtype=np.float)
       
        # if folderin and suffix have been provided, read cI file and calculate 
        # g2-1 by averaging cI
        if isinstance(suffix,str) and os.path.exists(folderin): 
            if suffix != 'cI_ts.dat' and suffix != 'cIcr_ts.dat' and \
               suffix != 'cI_norm_ts.dat' and suffix != 'cIcr_norm_ts.dat':
                raise NameError('Corr_func.__init__(): suffix must be one of\n'+
                                'cI_ts.dat, cIcr_ts.dat, cI_norm_ts.dat,'+\
                                'cIcr_norm_ts.dat\nsuffix = %s' % suffix )
            if folderin[-1] != '\\' and folderin[-1] != '/': folderin += '/'
            first = True
            for ir, rn in enumerate(self.__roi):  #loop over ROIs to process
                filein = 'ROI' + str(rn).zfill(ROIdigits) + suffix
                #get time delays, assign them to self:
                delays = sf.get_delays(folderin+filein)
                if first:
                    self.__n_tau = self.set_tau(delays)
                    first = False
                else:
                    if not np.allclose(delays,self.__tau):
                    #problem: all delay sets should be identical!
                        raise NameError('Corr_func.__init__(): delays should'+\
                            ' be identical for all xxxx_ts files!')            
                #update
                self.__sourcefile[ir] = folderin+filein
                #load cI
                cI = pd.read_csv(folderin+filein,sep="\t")
                #loop over time intervals over which g2-1 is to be averaged
                for it, tint in enumerate(self.__age):
                    t1 = tint[0]
                    t2 = tint[1]
                    reduced = cI.loc[cI['tsec'] >= t1].loc[cI['tsec'] <= t2]
                    cIchosen = np.asarray(reduced.iloc[:,3:],dtype = np.float)
                    g2 = np.nanmean(cIchosen,axis = 0)
                    np.copyto(self.__g2[it,ir],g2)            
        
        #variables for correcting the base line and normalizing the intercept
        #the correction/normalization is actually done in XXX according to
        # corrected g2-1 = [ [raw g2-1]-baseline ]/intercept
        # baseline and intercept are to be determined on the raw g2-1
        self.__bline = np.zeros(self.__n_roi)
        self.__intercept = np.ones((self.__n_age,self.__n_roi))
        #flag True/False if baseline correction and intercept normalization
        #already applied to g2-1
        self.__corrected = False  
        print(self.__g2)
        print(self.__tau)

        return            

        

    def deep_copy(self):
        """
        returns a deep copy of a Corr_func object, typically to normalize/
        correct for base line by applying bline_int_correct(self) to the 
        copy
        """
        return (copy.deepcopy(self))
        

    def bline_int_correct(self):
        """
        corrects g2-1 for base line and intercept, i.e. calculate a 'corrected'
        g2-1 by applying the following formula:
        corrected g2-1 = [ [raw g2-1]-baseline ]/intercept
        # baseline and intercept must have previously determined on the raw 
        g2-1, see XXX functions.
        Note: this formula implies that the intercept has been determined on
        [raw g2-1]-baseline, not just on [raw g2-1]

        """
        if self.__corrected == True:
            raise NameError('bline_int_correct(): base line and intercept'+\
                            ' corrections\nhave already been applied')
        for ir in range(self.__n_roi):
            self.__g2[:,ir,:] -= self.__bline[ir]
        for ia in range(self.__n_age):
            for ir in range(self.__n_roi):
                self.__g2[ia,ir,:] /= self.__intercept[ia,ir]
        self.__corrected = True  
        return        
        
        
        
    
    def get_age(self):
        """
        gets the sets of ages (list of tuples) of a Corr_func object
        """
        return (self.__age)

    def set_age_from_list(self,age):
        """
        assigns a set of ages ((tstart,tend) tuples) to a Corr_func object
        by passing a list of tuples
        raises an error if age.size != self.__n_age
        """
        if len(age) != (self.__n_age,): 
            raise NameError('set_age_from_list(): '+\
                            'mismatch in number of age tuples')
        for i, a in enumerate(age):
            self.__age[i] = a


    
    def set_age(self,tinit,tend,dt,nt,spacing='log',spacing_dt='ratio',\
                mindt=1.):
        """
        assigns a set of ages ((tstart,tend) tuples denoting time intervals) 
        to a Corr_func object based on the input parameters
        Makes sure that the duration of each time interval is at least mindt s 
        """
        
        #check input for consistency
        if tinit>=tend:
            raise NameError('set_age(): tinit must be < than tend')
        if dt <= 0:
            raise NameError('set_age(): dt must be > 0')
        if spacing =='log' and tinit < 1.:
            raise NameError('set_age(): tinit must be >= 1s when' + \
                            ' setting spacing=\'log\'\n')            
        if spacing_dt=='ratio' and dt <=1:
            raise NameError('set_age(): dt must be > 1 when' + \
                            ' setting spacing_dt=\'ratio\'')            
        if nt != self.__n_age:
            raise NameError('set_age(): nt must be same as self.__n_age = %d'\
                            % self.__n_age)
        
        
        if spacing !='log' and spacing != 'lin':spacing = 'log'
        if spacing_dt != 'ratio' and spacing_dt != 'const': spacing_dt='ratio' 
        if spacing == 'lin': 
            t1 = np.linspace(tinit,tend,nt)
        else:
            t1 = np.geomspace(tinit,tend,nt)
        if spacing_dt == 'const':
            t2 = t1+dt
        else:
            t2 = t1*dt
        for iage in range(self.__n_age):
            self.__age[iage] = (t1[iage],max(t2[iage],t1[iage]+mindt))




    def set_age_log_and_0(self,tinit,tend,dt,nt,spacing_dt='ratio',mindt=1.):
        """
        assigns a set of ages ((tstart,tend) tuples denoting time intervals) 
        to a Corr_func object based on the input parameters. 
        Same as running set_age() with log
        t spacing and nt-1 time intervals, to which the time interval starting
        at tinit=0 is prepended. 
        Makes sure that the duration of each time interval is at least mindt s 
        """
        
        #check input for consistency
        if tinit>=tend:
            raise NameError('set_age_log_and_0(): tinit must be < than tend')
        if dt <= 0:
            raise NameError('set_age_log_and_0(): dt must be > 0')
        if spacing_dt=='ratio' and dt <=1:
            raise NameError('set_age_log_and_0(): dt must be > 1 when' + \
                            ' setting spacing_dt=\'ratio\'')            
        if nt != self.__n_age:
            raise NameError('set_age(): nt must be same as self.__n_age = %d'\
                            % self.__n_age)
        
        
        if spacing_dt != 'ratio' and spacing_dt != 'const': spacing_dt='ratio' 
        t1 = np.geomspace(tinit,tend,nt-1)
        t1 = np.insert(t1,0,0.0)
        if spacing_dt == 'const':
            t2 = t1+dt
        else:
            t2 = t1*dt
        for iage in range(self.__n_age):
            self.__age[iage] = (t1[iage],max(t2[iage],t1[iage]+mindt))

    def get_roi(self):
        """
        gets the set of ROI numbers of a Corr_func object
        """
        return (self.__roi)

    def set_roi(self,roi):
        """
        assigns a set of ROI numbers to a Corr_func object
        raises an error if roi.size != self.__n_roi
        """
        if roi.size != self.__n_roi: 
            raise NameError('set_roi(): mismatch in number of ROIs')
        np.copyto(self.__roi,roi)
        
    def get_tau(self):
        """
        gets the set of time delays of a Corr_func object
        """
        return (self.__tau)
    
    def get_g2(self):
        """
        gets the set of g2 of a Corr_func object
        """
        return (self.__g2)
    
    def get_decaytime(self):
        """
        gets the set of g2 of a Corr_func object
        """
        return (self.__decaytime)

    def set_tau(self,tau):
        """
        assigns a set of time delays to a Corr_func object
        modifies Corr_func.g2 shape accordingly, if needed
        This function should be called before reading and averaging cI data
        to calculate g2 values
        """
        #check if cI data have already been read
        if self.__sourcefile[0] != '':
            raise NameError("set_tau() should be called before reading cI"+\
                            "file")
        ntau = tau.size
        shape_good = (self.__n_age,self.__n_roi,ntau)
        size_good = self.__n_age*self.__n_roi*ntau
        ds = size_good - self.__g2.size
        if ds > 0:
            self.__g2 = np.pad(self.__g2.ravel(),(0,ds),'constant')
        elif ds < 0:
            self.__g2 = self.__g2.ravel()[:size_good]
        self.__g2 = self.__g2.reshape(shape_good)
        self.__g2 *= np.nan    
                
        ds = tau.size - self.__n_tau
        if ds > 0:
            self.__tau = np.pad(self.__tau.ravel(),(0,ds),'constant')
        elif ds < 0:
            self.__tau = self.__tau.ravel()[:size_good]
        np.copyto(self.__tau,tau)
        self.__n_tau = self.__tau.size
        return self.__n_tau
                


    def set_bline(self,baseline):
        """
        sets the baseline value of a Corr_func object (self.__bline), using
        values passed in baseline, which should be a numpy float array of
        shape (self.__n_roi). Note: the base line may differ according to ROI,
        but should be the same for all ages
        """
        np.copyto(self.__bline,baseline)
        return        
        
   
    def set_intercept(self,intercept):
        """
        sets the intercept value of a Corr_func object (self.__intercept), 
        using values passed in intercept, a numpy float array of
        shape (self.__n_roi,self.__n_roi). Note: the intercept depends in 
        in general on both the age and the ROI
        """
        np.copyto(self.__intercept,intercept)
        return              
    
    
    def calc_bline(self, i_age:int, tau_bl: List[Tuple[float]],\
                   setbline: bool=True):
        """
        calculates the base line for each ROI, by averaging g2-1[i_age,:,:] 
        over the range of tau delays specified in tau_bl (list of tuples of 
        type (tau_init,tau_end))
        
        
        i_age: index of the age for which the tail of g2-1 is to be averaged to
        get the baseline 
        
        tau_bl: list of tuples (tau_init,tau_end) defining a range of tau
        over which g2-1 is to be averaged to get the baseline. Normally,
        one tuple per ROI (the base line may differ according to the ROI)
        If tau_bl contains less elements than the number of ROIs, the last 
        element will be replicated as needed. Hence, you may pass 
        tau_bl = [(tau_init,tau_end)] to set the same range of tau for all 
        ROIs
        
        if setbline==True: will update the Corr_func property self.__bline with
        the calculated baseline
        """
        
        l = len(tau_bl)
        dl = self.__n_roi - l
        if dl > 0: #fill age list up to number of ROIs using las list element
            tau_bl = tau_bl + tau_bl[-1]*dl
        baseline = np.zeros(self.__n_roi)
        for ir in range(self.__n_roi):
            tau_init  = tau_bl[ir][0]
            tau_end  = tau_bl[ir][1]
            goodtau = np.asarray((self.__tau >= tau_init) & \
                                 (self.__tau <= tau_end))
            baseline[ir] = np.nanmean(self.__g2[i_age,ir],where = goodtau)
        
        if setbline: np.copyto(self.__bline,baseline)
        return baseline
            
        

    def calc_intercept(self, tau_intercept: List[List[Tuple[float]]],\
                   setintercept: bool=True):
        """
        calculates the intercept for each age and ROI, by fitting 
        ln(g2-1[i_age,i_roi,tau_range]-baseline) to a straight line, where
        tau_range is a set of time delays indexes, normally designating the 
        shortest few delays, excluding tau=0 (reminder: for tau=0, g2-1 
        contains extra contributions due to self-correlated noise)
        
        tau_range will be calculated based on the info in tau_intercept
        
        tau_intercept is a list of lists of tuples, such that
        tau_intercept[i_age][i_roi] is a tuple of type (tau_init,tau_end), where
        tau_init,tau_end are the smallest/largest tau values that define the 
        fitting interval for the correlation function at age i_age and for ROI
        i_roi
        
        if seintercept==True: will update the Corr_func property 
        self.__intercept with
        the calculated baseline
        """

        
        intercept = np.zeros((self.__n_age,self.__n_roi))
        for ia in range(self.__n_age):
            for ir in range(self.__n_roi):
                tau_init  = tau_intercept[ia][ir][0]
                tau_end  = tau_intercept[ia][ir][1]
                goodtau = np.asarray((self.__tau >= tau_init) & \
                                     (self.__tau <= tau_end))
                if np.isclose(self.__tau[goodtau[0]],0): #exclude tau=0
                    goodtau = goodtau[1:]
                xfit = self.__tau[goodtau]
                yfit = self.__g2[ia,ir,goodtau]-self.__baseline[ir]
                goodfit = np.asarray(yfit>0)
                xfit = xfit[goodfit]
                yfit = np.ln(yfit[goodfit])
                p = np.polynomial.Polynomial.fit(xfit, yfit, 1)  #linear fit
                #gets the proper coefs from the fit, select the intercept coef
                lnint = p.convert(domain=(-1, 1)).coef[0]
                intercept[ia,ir] = np.exp(lnint)
        
        if setintercept: np.copyto(self.__intercept,intercept)
        return intercept        
    
    def g2_normalization(self,npoints,plot=True):
        
        for ia in range(self.__n_age):
            for ir in range(self.__n_roi):
                popt, pcov = curve_fit(sf.parab, self.__tau[1:npoints], self.__g2[ia][ir][1:npoints])
    
                if plot == True:
                    plt.figure()
                    plt.plot(self.__tau,self.__g2[ia][ir],marker='.',linestyle='')
                    plt.plot(self.__tau[1:npoints],sf.parab(self.__tau[1:npoints], *popt))
                    plt.xlim([self.__tau[0],self.__tau[npoints+2]])
                    plt.figure()
                    plt.semilogx(self.__tau,self.__g2[ia][ir]/ sf.parab(self.__tau[1:npoints], *popt)[0],marker='.',linestyle='')
                self.__g2[ia][ir] = self.__g2[ia][ir] / sf.parab(self.__tau[1:npoints], *popt)[0]
                

        return 
    
    
        
    def g2_from_cI(self,folderin,suffix,ROIdigits=4):
        """loads sets of cI data, calculates corresponding g2-1 averaged over
        time intervals pre-defined in the Corr_func object
        In principle, not to be used: rather create a new Corr_func object
        by passing the info on cI files to be loaded...
        """
        if suffix != 'cI_ts.dat' and suffix != 'cIcr_ts.dat' and \
           suffix != 'cI_norm_ts.dat' and suffix != 'cIcr_norm_ts.dat':
            raise NameError('g2_from_cI(): suffix must be one of\n'+
                            'cI_ts.dat, cIcr_ts.dat, cI_norm_ts.dat,'+\
                            'cIcr_norm_ts.dat\nsuffix = %s' % suffix )
        if folderin[-1] != '\\' and folderin[-1] != '/': folderin += '/'
        for ir, rn in enumerate(self.__roi):  #loop over ROIs to process
            filein = 'ROI' + str(rn).zfill(ROIdigits) + suffix
            self.__sourcefile[ir] = folderin+filein
            #get time delays, assign them to self:
            delays = sf.get_delays(folderin+filein)
            self.set_tau(delays)
            #load cI
            cI = pd.read_csv(folderin+filein,sep="\t")
            #loop over time intervals over which g2-1 is to be averaged
            for it, tint in enumerate(self.__age):
                t1 = tint[0]
                t2 = tint[1]
                reduced = cI.loc[cI['tsec'] >= t1].loc[cI['tsec'] <= t2]
                cIchosen = np.asarray(reduced.iloc[:,3:],dtype = np.float)
                g2 = np.nanmean(cIchosen,axis = 0)
                np.copyto(self.__g2[it,ir],g2)
            

    def g2_save(self,folderout,classify='',ROIdigits=4,verbose=True):
        """saves to file g2-1 stored in Corr_func object
        classify = 'by ROI' --> one file per ROI, within file one curve per age
        classify = 'by age' --> one file per age, within file one curve per ROI
        classify = '' --> as both 'by ROI' and 'by age'
        Creates folderout if needed
        Saves also the following in folderout
        - corr_func.pickled: the Corr_func object as a pickled binary file 
        - corr_func_info.txt: file with some info on the Corr_func object
        """
        folderout = sf.prepare_folder(folderout)
        #issue warning if g2 data already exist in folderout
        if verbose and os.path.isfile(folderout+'corr_func.pickle'):
            msg = 'WARNING in g2_save():\n%s\n already contains g2-1 data.\n' % folderout 
            msg += 'Do you want to continue and overwrite the existing data (Y/N)?\n'
            ok = input(msg)
            if ok.lower()[0] != 'y':
                print('g2_save() aborted')
                return
        
        pos = self.__sourcefile[0].find('cI')
        suffix = self.__sourcefile[0][pos:-3] + 'dat'

        doboth = False
        #output both kinds of files unless explicitly specified
        if classify != 'by ROI' and classify != 'by age': doboth = True
        
        #output one file per ROI
        if doboth: classify = 'by ROI'
        if classify == 'by ROI':
            if verbose: print('\nsaving g2-1, one file per ROI')
            #header common to all files (except first col)
            colnames = []
            for iage in range(self.__n_age):
                colnames.append('tw%.2e-%.2e' % \
                            (self.__age[iage][0],self.__age[iage][1]))
            #loop over ROIs
            for iroi in range(self.__n_roi):
                foutname = 'g2_ROI'+ str(self.__roi[iroi]).zfill(ROIdigits)+\
                    '_' + suffix
                data = self.__g2[:,iroi,:].transpose()
                df = pd.DataFrame(data,columns = colnames,dtype=float)
                df.insert(0,'tau.s',self.__tau)
                df.to_csv(folderout+foutname,sep='\t',index=False,\
                          na_rep='nan',float_format="%.4f")
            
        #output one file per age
        if doboth: classify = 'by age' 
        if classify == 'by age':
            if verbose: print('\nsaving g2-1, one file per age')
            #header common to all files
            colnames = []
            for iroi in range(self.__n_roi):
                colnames.append('ROI'+ str(self.__roi[iroi]).zfill(ROIdigits))
            #loop over ages
            for iage in range(self.__n_age):
                foutname = 'g2_age_%.2e-%.2e' % \
                            (self.__age[iage][0],self.__age[iage][1])
                foutname = foutname + '_' + suffix
                data = self.__g2[iage,:,:].transpose()
                df = pd.DataFrame(data,columns=colnames,dtype=float)
                df.insert(0,'tau.s',self.__tau)
                df.to_csv(folderout+foutname,sep='\t',index=False,\
                          na_rep='nan',float_format="%.4f")
            
        #output Corr_func object as binary pickled file and recap info on
        #object in Corr_func_info.txt
        with open(folderout+'corr_func.pickle', 'wb') as fout:
            pk.dump(self,fout)
        with open(folderout+'corr_func_info.txt', 'w') as fout:
            text = 'Info on correlation function(s) saved in this folder\n' +\
                    '%d ages (t1,t2) in s:\n' % self.__n_age
            fout.write(text)
            for i in range(self.__n_age): fout.write(str(self.__age[i])+'\n')
            fout.write('\n%d ROI(s), source cI file(s) are:\n' % self.__n_roi)
            for i in range(self.__n_roi): 
                fout.write(self.__sourcefile[i]+'\n')
            fout.write('\n%d delays tau (in s):\n' % self.__n_tau)
            for i in range(self.__n_tau): 
                fout.write(str(self.__tau[i])+'\n')
        return     
        
        
        

    def g2_plot(self,folderout='',byROI=True,ROIdigits=4,fignum=None):
        """ plots g2-1
        if byROI = True: one subplot for each ROI, 
            within a subplot one curve per age
        if byROI= False: one subplot for each age, 
            within a subplot one curve per ROI
        if folderout != '', the figure will be saved in folderout (as pdf and 
        with a standard name). Creates folderout if needed
        Returns figure and axes (matplotlib objects)
        """
        if byROI: 
            nsubplots = self.__n_roi
            lblprefix = '$t_w=$ '
        else : 
            lblprefix = 'ROI '
            nsubplots = self.__n_age
        ncols = 2 
        nrows = int(nsubplots/ncols+0.5)
        fig,ax = plt.subplots(nrows,ncols,figsize=(10,5*nrows),num=fignum)
        ax = ax.flatten()
        #loop over subplots
        for isub in range(nsubplots):
            if byROI:    
                #plot by ROI: each plot shows curves at various ages
                ax[isub].set_title('ROI %d' % self.__roi[isub])
                #loop over ages, add curve to plot
                for iage in range(self.__n_age):
                    if isub==nsubplots-1:
                        lbl = lblprefix + '%.3e s' % self.__age[iage][0]
                    else:
                        lbl=''
                    ax[isub].plot(self.__tau,self.__g2[iage,isub],label=lbl)
            
            if not byROI:    
                #plot by age: each plot shows curves for various ROIs
                ax[isub].set_title('Age %.2e-%.2e s' % \
                       (self.__age[isub][0],self.__age[isub][1]) )
                #loop over ROIs, add curve to plot
                for iroi in range(self.__n_roi):
                    if isub==nsubplots-1:
                        lbl = lblprefix + '%d' % self.__roi[iroi]
                    else:
                        lbl=''
                    ax[isub].plot(self.__tau,self.__g2[isub,iroi],label=lbl)

            #format axes etc. of subplot
            ax[isub].set_xscale('log')
            ax[isub].set_xlabel('$\\tau$ (s)')
            ax[isub].set_ylabel('$g_2(\\tau)-1$')
            
        #save figure if needed
        if folderout != '':
            folderout = sf.prepare_folder(folderout)
            pos = self.__sourcefile[0].find('cI')
            suffix = self.__sourcefile[0][pos:-3] + 'pdf'
            if byROI:
                figfile = 'g2_ROI'+ str(self.__roi[0]).zfill(ROIdigits) +'-' +\
                    str(self.__roi[-1]).zfill(ROIdigits) + '_' + suffix
            if not byROI:
                figfile = 'g2_age_%.2e-%.2e_' % \
                            (self.__age[0][0],self.__age[-1][0])
                figfile += ('_' + suffix)
        #add figure title, figure legend (legend is the same for all subplots!)
            fig.suptitle('This file: '+folderout+figfile +'\nData from '+ \
                         self.__sourcefile[0] + ' etc.')
            fig.legend()
            fig.tight_layout()
        #save figure
            fig.savefig(folderout+figfile)
            os.startfile(folderout+figfile)   # open the figure in the default pdf reader 
        else:
            #fig.suptitle('Data from '+ self.__sourcefile[0] + ' etc.')
            fig.legend()
            fig.tight_layout()

        return fig,ax  
    
    def DecaytimeFromArea(self,folderout='',clean=True):
        
        if clean:
            self.__decaytime = np.zeros(shape=(self.__n_age,self.__n_roi,1),\
                           dtype=np.float)
            
        for isub in range(self.__n_roi):
            #loop over ages, add curve to plot
            for iage in range(self.__n_age):
                I,stracazzo = sf.SFintegration(self.__tau,self.__g2[iage,isub],self.__tau[1],self.__tau[-1])
                np.copyto(self.__decaytime[iage,isub],I)
            
            
        return

            
            
        
