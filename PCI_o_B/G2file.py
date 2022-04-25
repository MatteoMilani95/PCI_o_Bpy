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
    def __init__(self,FolderName,Timepulse = False):
        super().__init__()
        
        self.FolderName = FolderName
        self.CI = []
        self.nROI = []
        self.g2 = []
        self.g2var = []
        self.tau = []
        self.taug2 = []
        self.scatt_angle = []
        self.scatt_angle_exp = []
        self.Center = 0
        self.decaytime1 = []
        self.decaytime1err = []
        self.decaytime2 = []
        self.decaytime2err = []
        self.Timepulse = Timepulse
        
        
    def __str__(self):
        str_res  = '\n|---------------|'
        str_res += '\n| G2 class:    '
        str_res += '\n|--------------------+--------------------|'
        str_res += '\n| filelist             : ' + str(self.ROIfilelist)
        str_res += '\n| folder               : ' + str(self.FolderName) 
        str_res += '\n| number of ROIs       : ' + str(self.nROI) 
        str_res += '\n| ROIs size            : ' + str(self.GetROIsize()) + ' px'
        str_res += '\n| x for theta(x)= 90°  : ' + str(self.Center) + 'px'
        str_res += '\n| Radius bead          : ' + str(self.Center) +'px'
        #str_res += '\n| Window of interest top : ' + str(self.GetWINDOWtop()) + ' px'
        str_res += '\n|--------------------+--------------------|'
        return str_res
    
    
    def G2Calculation(self,CI,nROI,tau,*args):
        '''
        

        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        self.CI = CI
        self.nROI = nROI
        self.tau = tau.copy()
        
        g2_inf=[]
        var_inf=[]
        
        if args:
            if len(args) !=2:
                print('give 2 int start and stop')
                return
            else:
                if self.Timepulse == True:
                    for i in range(self.nROI):
                        g2_inf.append(self.CI[i].iloc[int(args[0]/2):int(args[1]/2)].mean(axis = 0))
                        var_inf.append(self.CI[i].iloc[int(args[0]/2):int(args[1]/2)].var(axis = 0))

                else:
                    for i in range(self.nROI):
                        g2_inf.append(self.CI[i].iloc[args[0]:args[1]].mean(axis = 0))
                        var_inf.append(self.CI[i].iloc[args[0]:args[1]].var(axis = 0))
                    
        else:
            for i in range(self.nROI):
                g2_inf.append(self.CI[i].mean(axis = 0))
                var_inf.append(self.CI[i].var(axis = 0))
                
        g2=[]
        var=[]
        
        for i in range(self.nROI):    
            g2.append(g2_inf[i].replace([np.inf, -np.inf], 0))
            var.append(var_inf[i].replace([np.inf, -np.inf], 0))
        
            
        remove_nan = []
        remove_nan_v = []
        bho = []
        
        for i in range(self.nROI):
            nan_elems = g2[i].isnull()
            nan_elems_v = var[i].isnull()
            remove_nan.append(g2[i][~nan_elems])
            remove_nan_v.append(var[i][~nan_elems_v])
            
            a = pd.Series([0],index=[remove_nan[i].index[-1]])
            while len(remove_nan_v[i])<len(remove_nan[i]):
                remove_nan_v[i] = remove_nan_v[i].append(a)
                
            bho.append(remove_nan_v[i].replace(0,remove_nan_v[i][2:].max(axis=0) ))
        
        
                
        
        for i in range(self.nROI):
            self.g2.append(remove_nan[i][2:])
            #self.g2 = [x for x in self.g2 if str(x) != 'nan']
            self.g2var.append(bho[i][2:])
            
            
        
      
        
        if len(self.tau)>len(self.g2[0]):
            while len(self.tau)>len(self.g2[0]):
                self.tau.pop()
       
            
        for i  in range(self.nROI):
            self.taug2.append(self.tau)
            
            
        return
    
        for i  in range(self.nROI):
            self.taug2.append(self.tau)
    
    
    
    def G2Set0Baseline(self):

        g2baseline = []
         
        for i in range(self.nROI):
            
            g2baseline.append(self.g2[i][-1])
            
     
            
        for i in range(self.nROI):
            
            for j in range(len(self.g2[i])):
                if self.g2[i][j] > 0:
                    self.g2[i][j] = self.g2[i][j] - g2baseline[i] 
            
                    
                

            
        return 
    
   
        
    def fitG2(self,function,variables):
        
        outparam = []
        
        for i in range(self.nROI):
            outparam.append(curve_fit(function,  np.asarray(self.taug2[i]), np.asarray(self.g2[i]), variables, np.asarray(self.g2var[i]) ))

        return outparam
    
    
    def FitSingleDecaytime(self,variables,plot):
        
        self.decaytime1 = []
        self.decaytime1err = []
        
        fitted_curve = []
        
        outparam = self.fitG2(sf.SingExp,variables)
        
        folder_fit_graphs = self.FolderName + '\\fit_graphs'
        
        try:
            os.mkdir(folder_fit_graphs)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        for i in range(self.nROI):
             fitted_curve.append(sf.SingExp( np.asarray(self.taug2[i]), outparam[i][0][0], outparam[i][0][1], outparam[i][0][2]))
             
    
        for i in range(self.nROI):
            self.decaytime1.append(outparam[i][0][1])
            self.decaytime1err.append(2*np.sqrt(outparam[i][1][1][1]))
        
        goodness_fit = [] 
        for i in range(self.nROI):
            goodness_fit.append( np.sum( ( ( np.asarray(fitted_curve[i])-np.asarray(self.g2[i]))  )**2 / np.asarray(fitted_curve[i]) ) )
    
        np.logspace(5*1e-5,self.taug2[i][-1],100)
        
        
        curve_for_plot = []
        time_for_plot = []
        
        for i in range(self.nROI):
            time_for_plot.append(np.logspace(-5,-2,100))
            curve_for_plot.append(sf.SingExp( time_for_plot[i], outparam[i][0][0], outparam[i][0][1], outparam[i][0][2]))
             
        
        if plot == True:
            fig = plt.figure()
            ax = plt.axes() 
            for i in range(self.nROI):
                ax.set_xscale('log')
                #plt.yscale('log')
                ax.plot(self.taug2[i],np.asarray(self.g2[i])/np.asarray(fitted_curve[i][0]),linestyle='',marker='o',label= 'ROI ' + str(i+1))
                ax.plot(time_for_plot[i],np.asarray(curve_for_plot[i])/np.asarray(curve_for_plot[i][0]),'-')
                #print(np.asarray(self.g2[i])/np.asarray(fitted_curve[i][0]))
                ax.set_xlabel(r'$\tau$' + str(' ') + ' [s]',fontsize=14)
                ax.set_ylabel(r'$g_2-1$',fontsize=14)
                ax.tick_params(bottom=True, top=True, left=True, right=False)
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
                ax.tick_params(axis="x", direction="in")
                ax.tick_params(axis="y", direction="in")
                ax.tick_params(axis='x', which='minor', direction="in")
                ax.set_ylim([-0.1, 1.1])
                ax.legend(loc='lower left')
                #plt.grid(True)
                plt.savefig(folder_fit_graphs+'\\g2_ROI'+str(i+1).zfill(4)+'.png')
                
                paper = True
                
            if paper == True:
                fig = plt.figure()
                ax = plt.axes() 
    
                ax.set_xscale('log')
                #plt.yscale('log')
                ax.plot(self.taug2[0],np.asarray(self.g2[0])/np.asarray(fitted_curve[0][0]),linestyle='',color='black',marker='o',label= ' x = 1.6 mm' )
                ax.plot(time_for_plot[0],np.asarray(curve_for_plot[0])/np.asarray(curve_for_plot[0][0]),'-',color='black')
                    
                ax.plot(self.taug2[2],np.asarray(self.g2[2])/np.asarray(fitted_curve[2][0]),linestyle='',color='darkslategray',marker='s',label= ' x = 0.8 mm')
                ax.plot(time_for_plot[2],np.asarray(curve_for_plot[2])/np.asarray(curve_for_plot[2][0]),'-',color='darkslategray')
                    
                ax.plot(self.taug2[4],np.asarray(self.g2[4])/np.asarray(fitted_curve[4][0]),linestyle='',color='darkcyan',marker='h',label= ' x = 0.0 mm')
                ax.plot(time_for_plot[4],np.asarray(curve_for_plot[4])/np.asarray(curve_for_plot[4][0]),'-',color='darkcyan')
                    
                ax.plot(self.taug2[6],np.asarray(self.g2[6])/np.asarray(fitted_curve[6][0]),linestyle='',color='deepskyblue',marker='>',label= ' x = - 0.8 mm')
                ax.plot(time_for_plot[6],np.asarray(curve_for_plot[6])/np.asarray(curve_for_plot[6][0]),'-',color='deepskyblue')
                    
                ax.plot(self.taug2[8],np.asarray(self.g2[8])/np.asarray(fitted_curve[8][0]),linestyle='',color='lightskyblue',marker='*',label= ' x = - 1.6 mm')
                ax.plot(time_for_plot[8],np.asarray(curve_for_plot[8])/np.asarray(curve_for_plot[8][0]),'-',color='lightskyblue')
                #print(np.asarray(self.g2[i])/np.asarray(fitted_curve[i][0]))
                ax.set_xlabel(r'$\tau$' + str(' ') + ' [s]',fontsize=14)
                ax.set_ylabel(r'$g_2-1$',fontsize=14)
                ax.tick_params(bottom=True, top=True, left=True, right=False)
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
                ax.tick_params(axis="x", direction="in")
                ax.tick_params(axis="y", direction="in")
                ax.tick_params(axis='x', which='minor', direction="in")
                ax.set_ylim([-0.1, 1.1])
                ax.legend(loc='lower left')
                #plt.grid(True)
                plt.savefig('C:\\Users\\Matteo\\Desktop\\PHD\\paper_1\\g2picture.pdf')
        else:
            return
        
        return
    
    def FitDoubleDecaytime(self,variables,plot):
        
        self.decaytime1 = []
        self.decaytime1err = []
        self.decaytime2 = []
        self.decaytime2err = []
        
        fitted_curve = []
        
        outparam = self.fitG2(sf.DoubleExp,variables)
        
        folder_fit_graphs = self.FolderName + '\\fit_graphs'
        
        try:
            os.mkdir(folder_fit_graphs)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        for i in range(self.nROI):
             fitted_curve.append(sf.DoubleExp( np.asarray(self.taug2[i]), outparam[i][0][0], outparam[i][0][1], outparam[i][0][2], outparam[i][0][3], outparam[i][0][4]))
             
    
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
                plt.errorbar(self.taug2[i],self.g2[i],yerr=self.g2var[i],fmt='o',label='chi = ' + str(goodness_fit[i]) + '\n' + 'baseline = ')
                plt.plot(self.taug2[i],fitted_curve[i],'-.')
                plt.xlabel('tau  [s]')
                plt.ylabel('g2-1')
                plt.title('g2_ROI'+str(i+1).zfill(4))
                plt.ylim([-0.1, 1.1])
                plt.legend(loc='upper right')
                #plt.grid(True)
                plt.savefig(folder_fit_graphs+'\\g2_ROI'+str(i+1).zfill(4)+'.png')
        else:
            return
        
        return
    
    def FitSingleStretchDecaytime(self,variables,plot):
        
        self.decaytime1 = []
        self.decaytime1err = []
        self.decaytime2 = []
        self.decaytime2err = []
            
        fitted_curve = []
            
        outparam = []
        
        for i in range(self.nROI):
            outparam.append(curve_fit(sf.SingleStretchExp,  np.asarray(self.taug2[i]), np.asarray(self.g2[i]), variables, np.asarray(self.g2var[i]),bounds=([-np.inf,-np.inf,-0.001,0.1],[np.inf,np.inf,0.001,10]) ))

                
        folder_fit_graphs = self.FolderName + '\\fit_graphs'
            
        try:
            os.mkdir(folder_fit_graphs)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
            
        for i in range(self.nROI):
            fitted_curve.append(sf.SingleStretchExp( np.asarray(self.taug2[i]), outparam[i][0][0], outparam[i][0][1], outparam[i][0][2] , outparam[i][0][3]))
                 
        
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
                plt.errorbar(self.taug2[i],self.g2[i],yerr=self.g2var[i],fmt='o',label='chi = ' + str(goodness_fit[i]) + '\n' + 'baseline = ' + str(outparam[i][0][2]) + '\n' + 'decaytime1 = ' + str(outparam[i][0][1])+  '\n' + 'beta = ' + str(outparam[i][0][3]))
                plt.plot(self.taug2[i],fitted_curve[i],'-.')
                plt.xlabel('tau  [s]')
                plt.ylabel('g2-1 ')
                plt.title('g2_ROI'+str(i+1).zfill(4))
                plt.ylim([-0.1, 1.1])
                plt.legend(loc='lower left')
                #plt.grid(True)
                plt.savefig(folder_fit_graphs+'\\g2_ROI'+str(i+1).zfill(4)+'.png')
        else:
            return
        
        return
    
    def FitDoubleStretchDecaytime(self,variables,plot):
        
        self.decaytime1 = []
        self.decaytime1err = []
        self.decaytime2 = []
        self.decaytime2err = []
            
        fitted_curve = []
            
        outparam = []
        
        for i in range(self.nROI):
            outparam.append(curve_fit(sf.DoubleStretchExp,  np.asarray(self.taug2[i]), np.asarray(self.g2[i]), variables, np.asarray(self.g2var[i]),bounds=([-np.inf,-np.inf,-np.inf,-np.inf,-0.001,0.5,0.5],[np.inf,np.inf,np.inf,np.inf,0.001,10,10]) ))

                
        folder_fit_graphs = self.FolderName + '\\fit_graphs'
            
        try:
            os.mkdir(folder_fit_graphs)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
            
        for i in range(self.nROI):
            fitted_curve.append(sf.DoubleStretchExp( np.asarray(self.taug2[i]), outparam[i][0][0], outparam[i][0][1], outparam[i][0][2] , outparam[i][0][3], outparam[i][0][4] , outparam[i][0][5], outparam[i][0][6]))
                 
        
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
                plt.errorbar(self.taug2[i],self.g2[i],yerr=self.g2var[i],fmt='o',label='chi = ' + str(goodness_fit[i]) + '\n' + 'baseline = ' + str(outparam[i][0][4]) + '\n' + 'decaytime = ' + str(outparam[i][0][1]) + '\n' + 'beta1 = ' + str(outparam[i][0][5]) + '\n'+ 'beta2 = ' + str(outparam[i][0][6]))
                plt.plot(self.taug2[i],fitted_curve[i],'-.')
                plt.xlabel('tau  [s]')
                plt.ylabel('g2-1 ')
                plt.title('g2_ROI'+str(i+1).zfill(4))
                plt.ylim([-0.1, 1.1])
                plt.legend(loc='lower left')
                #plt.grid(True)
                plt.savefig(folder_fit_graphs+'\\g2_ROI'+str(i+1).zfill(4)+'.png')
        else:
            return
        
        return
    
    def FitStretchDecaytime(self,variables,plot):
        
        self.decaytime1 = []
        self.decaytime1err = []

            
        fitted_curve = []
            
        outparam = []
        
        for i in range(self.nROI):
            outparam.append(curve_fit(sf.StretchExp,  np.asarray(self.taug2[i]), np.asarray(self.g2[i]), variables, np.asarray(self.g2var[i]),bounds=([-np.inf,-np.inf,-0.001,0.1],[np.inf,np.inf,0.001,0.9]) ))

                
        folder_fit_graphs = self.FolderName + '\\fit_graphs'
            
        try:
            os.mkdir(folder_fit_graphs)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
            
        for i in range(self.nROI):
            fitted_curve.append(sf.StretchExp( np.asarray(self.taug2[i]), outparam[i][0][0], outparam[i][0][1], outparam[i][0][2] , outparam[i][0][3]))
                 
        
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
                plt.errorbar(self.taug2[i],self.g2[i],yerr=self.g2var[i],fmt='o',label='chi = ' + str(goodness_fit[i]) + '\n' + 'baseline = ' + str(outparam[i][0][2]) + '\n' + 'decaytime = ' + str(outparam[i][0][1]) + '\n' + 'beta = ' + str(outparam[i][0][3]))
                plt.plot(self.taug2[i],fitted_curve[i],'-.')
                plt.xlabel('tau  [s]')
                plt.ylabel('g2-1 ')
                plt.title('g2_ROI'+str(i+1).zfill(4))
                plt.ylim([-0.1, 1.1])
                plt.legend(loc='lower left')
                #plt.grid(True)
                plt.savefig(folder_fit_graphs+'\\g2_ROI'+str(i+1).zfill(4)+'.png')
        else:
            return
        
        return
    
    def FitTripleDecaytime(self,variables,plot):
        
        self.decaytime1 = []
        self.decaytime1err = []
        self.decaytime2 = []
        self.decaytime2err = []
        self.decaytime3 = []
        self.decaytime3err = []
        
        fitted_curve = []
        
        outparam = self.fitG2(sf.TripleExp,variables)
        
        folder_fit_graphs = self.FolderName + '\\fit_graphs'
        
        try:
            os.mkdir(folder_fit_graphs)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        for i in range(self.nROI):
             fitted_curve.append(sf.TripleExp( np.asarray(self.taug2[i]), outparam[i][0][0], outparam[i][0][1], outparam[i][0][2], outparam[i][0][3], outparam[i][0][4], outparam[i][0][5], outparam[i][0][6]))
             
    
        for i in range(self.nROI):
            self.decaytime1.append(outparam[i][0][1])
            self.decaytime1err.append(2*np.sqrt(outparam[i][1][1][1]))
            self.decaytime2.append(outparam[i][0][3])
            self.decaytime2err.append(2*np.sqrt(outparam[i][1][3][3]))
            self.decaytime3.append(outparam[i][0][5])
            self.decaytime3err.append(2*np.sqrt(outparam[i][1][5][5]))
        
        goodness_fit = [] 
        for i in range(self.nROI):
            goodness_fit.append( np.sum( ( ( np.asarray(fitted_curve[i])-np.asarray(self.g2[i]))  )**2 / np.asarray(fitted_curve[i]) ) )
    
        if plot == True:
            for i in range(self.nROI):
                plt.figure() 
                plt.xscale('log')
                plt.errorbar(self.taug2[i],self.g2[i],yerr=self.g2var[i],fmt='o',label='chi = ' + str(goodness_fit[i]) + '\n' + 'baseline = ')
                plt.plot(self.taug2[i],fitted_curve[i],'-.')
                plt.xlabel('tau  [s]')
                plt.ylabel('g2-1')
                plt.title('g2_ROI'+str(i+1).zfill(4))
                plt.ylim([-0.1, 1.1])
                plt.legend(loc='upper right')
                #plt.grid(True)
                plt.savefig(folder_fit_graphs+'\\g2_ROI'+str(i+1).zfill(4)+'.png')
        else:
            return
        
        return
    
    def G2Show(self,which_ROI):
        
        folder_G2_graphs = self.FolderName + '\\G2_graphs'
        
        try:
            os.mkdir(folder_G2_graphs)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        if isinstance(which_ROI, str):
            plt.figure()
            for i in range(self.nROI): 
                 
                plt.xscale('log')
                plt.errorbar(self.taug2[i],self.g2[i],yerr=self.g2var[i],fmt='o',label='ROI '+str(i+1),marker = ".")
                plt.xlabel('tau  [s]')
                plt.ylabel('g2-1')
                plt.ylim([-0.1, 1.1])
            plt.title('g2_ROI')
            plt.legend(loc='lower left',fontsize='xx-small')
            plt.savefig(folder_G2_graphs+'\\g2_ROI.png', dpi=300)
        else:
            plt.figure() 
            plt.xscale('log')
            plt.errorbar(self.taug2[which_ROI-1],self.g2[which_ROI-1],yerr=self.g2var[which_ROI-1],fmt='o')
            plt.xlabel('tau  [s]')
            plt.ylabel('g2-1')
            plt.ylim([-0.1, 1.1])
            plt.title('g2_ROI'+str(which_ROI).zfill(4))
            
        paper = False
        if paper == True:
            fig = plt.figure()
            ax = plt.axes() 
    
            ax.set_xscale('log')
                #plt.yscale('log')
            ax.plot(self.taug2[6],self.g2[6],marker='o',color='black',label='x= 0.00')
                    
            ax.plot(self.taug2[5],self.g2[5],marker='s',color='darkgreen',label='x= 0.27')
                    
            ax.plot(self.taug2[4],self.g2[4],marker='d',color='green',label='x= 0.54')
                    
            ax.plot(self.taug2[3],self.g2[3],marker='p',color='olivedrab',label='x= 0.81')
                    
            ax.plot(self.taug2[2],self.g2[2],marker='>',color='limegreen',label='x= 1.08')
            
            ax.plot(self.taug2[1],self.g2[1],marker='*',color='yellowgreen',label='x= 1.35')
            
            ax.plot(self.taug2[0],self.g2[0],marker='+',color='greenyellow',label='x= 1.62')
                #print(np.asarray(self.g2[i])/np.asarray(fitted_curve[i][0]))
            ax.set_xlabel(r'$\tau$' + str(' ') + ' [s]',fontsize=14)
            ax.set_ylabel(r'$g_2-1$',fontsize=14)
            ax.tick_params(bottom=True, top=True, left=True, right=False)
            ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
            ax.tick_params(axis="x", direction="in")
            ax.tick_params(axis="y", direction="in")
            ax.tick_params(axis='x', which='minor', direction="in")
            ax.set_ylim([-0.1, 1.1])
            ax.legend(loc='lower left')
                #plt.grid(True)
            plt.savefig('C:\\Users\\Matteo\\Desktop\\PHD\\paper_1\\g2picture_shell.pdf')
        return
    
    
    def G2ShowFlactuations(self,which_ROI):
        
        folder_G2_graphs = self.FolderName + '\\G2_graphs'
        
        try:
            os.mkdir(folder_G2_graphs)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        if isinstance(which_ROI, str):
            plt.figure()
            for i in range(self.nROI): 
                 
                plt.xscale('log')
                plt.plot(self.taug2[i],self.g2var[i],marker='o',label='ROI '+str(i+1))
                plt.xlabel('tau  [s]')
                plt.ylabel('var')
                #plt.ylim([-0.1, 1.1])
            plt.title('varg2_ROI')
            plt.legend(loc='lower left',fontsize='xx-small')
            plt.savefig(folder_G2_graphs+'\\varg2_ROI.png', dpi=300)
        else:
            plt.figure() 
            plt.xscale('log')
            plt.plot(self.taug2[which_ROI-1],self.g2var[which_ROI-1],marker='o')
            plt.xlabel('tau  [s]')
            plt.ylabel('g2-1')
            #plt.ylim([-0.1, 1.1])
            plt.title('g2_ROI'+str(which_ROI).zfill(4))
        return
    
    def G2CutBaseLine(self,nPoints,ROI=0):
        
        if ROI==0:
            cut = []
            for i in range(nPoints):
                cut.append(self.g2[0].index[-i-1])
            
            
                    
            
            for i in range(self.nROI):
                self.g2[i].drop(cut,inplace=True)
                self.g2var[i].drop(cut,inplace=True)
                self.taug2[i] = self.taug2[i][:len(self.taug2[i])-nPoints]
                

        else:
            cut = []
            for i in range(nPoints):
                cut.append(self.g2[ROI-1].index[-i-1])
                
            self.g2[ROI-1].drop(cut,inplace=True)
            self.g2var[ROI-1].drop(cut,inplace=True)
            
            self.taug2[ROI-1] = self.taug2[ROI-1][:len(self.taug2[ROI-1])-nPoints]   
       
        
        return 
    
    def G2Normalize(self,ROI=0):
        if ROI==0:
            for i in range(self.nROI):
                self.g2[i] = self.g2[i] / self.g2[i][1]
                
        else:
              self.g2[ROI-1] = self.g2[ROI-1] / self.g2[ROI-1][1]
             
             
        
        return
    
    def G2CutIntercept(self,nPoints,ROI=0):
        
        if ROI==0:
            cut = []
            for i in range(nPoints):
                cut.append(self.g2[0].index[i])
            
            for i in range(self.nROI):
                self.g2[i].drop(cut,inplace=True)
                self.g2var[i].drop(cut,inplace=True)
                self.taug2[i] = self.taug2[i][nPoints:] 
            
            
            '''
            for i in range(nPoints):
                del self.tau[0]
            '''
                
        else:
            cut = []
            for i in range(nPoints):
                cut.append(self.g2[ROI-1].index[-i-1])
                
            self.g2[ROI-1].drop(cut,inplace=True)
            self.g2var[ROI-1].drop(cut,inplace=True)
                
            self.taug2[ROI-1] = self.taug2[ROI-1][nPoints:]   
            
            
            
        
        
        return 
    
    def G2CutPoint(self,Point,nPoints,ROI=0):
        
        if ROI==0:
            cut = []
            
            for i in range(nPoints):
                cut.append(self.g2[0].index[Point-1+i])
                
                
            
            
            
            for i in range(self.nROI):
                self.g2[i].drop(cut,inplace=True)
                self.g2var[i].drop(cut,inplace=True)
                
                self.taug2[i] = self.taug2[i][:Point-1]+self.taug2[i][Point-1+nPoints:]
            
                          

        else:
            cut = []
            
            for i in range(nPoints):
                cut.append(self.g2[ROI-1].index[Point-1+i])
                print(cut)
                print('figa')
            
            
            self.g2[ROI-1].drop(cut,inplace=True)
            self.g2var[ROI-1].drop(cut,inplace=True)
                
            self.taug2[ROI-1] = self.taug2[ROI-1][:Point-1]+self.taug2[ROI-1][Point-1+nPoints:]
            
        return 
    
   
    
    def G2AreaDecaytime(self):
        
        self.decaytime1 = []
        self.decaytime1err = []
        
        folder_fit_graphs = self.FolderName + '\\fit_graphs'
        
        try:
            os.mkdir(folder_fit_graphs)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')

        for i in range(self.nROI):

            I,stracazzo = sf.SFintegration(self.taug2[i][1:],self.g2[i][1:],self.taug2[i][1],self.taug2[i][-1])
            self.decaytime1.append(I)
            self.decaytime1err.append(stracazzo)
            
            
            formatted_float = "{:.2f}".format(self.decaytime1[i] )
            plt.figure() 
                
            plt.xscale('log')
            plt.errorbar(self.taug2[i],self.g2[i],yerr=self.g2var[i],fmt='o',label= 'decaytime = '+ formatted_float + ' s')
   
            plt.xlabel('tau  [s]')
            plt.ylabel('g2-1 ')
            plt.title('g2_ROI'+str(i+1).zfill(4))
            plt.ylim([-0.1, 1.1])
            plt.legend(loc='lower left')
                #plt.grid(True)
            plt.savefig(folder_fit_graphs+'\\g2_ROI'+str(i+1).zfill(4)+'.png')
            
        
        
        return
    
    def SaveG2(self):
        
        folder_G2_Processed = self.FolderName + '\\processed_G2\\'
        
        try:
            os.mkdir(folder_G2_Processed)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        for i in range(self.nROI):
            self.g2[i].to_csv(folder_G2_Processed + 'ROI' + str(i+1).zfill(4) + 'g2.dat',sep='\t',index=False, header=None,na_rep='NaN')
            self.g2var[i].to_csv(folder_G2_Processed + 'ROI' + str(i+1).zfill(4) + 'varg2.dat',sep='\t',index=False, header=None,na_rep='NaN')
            pd.Series(self.taug2[i]).to_csv(folder_G2_Processed + 'ROI' + str(i+1).zfill(4) + 'taug2.dat',sep='\t', header=None,index=False,na_rep='NaN')
        return
        
    
    def UploadG2(self,nROI):
        
        self.nROI = nROI
        

        G2filelist = []
        tauG2filelist = []
        varG2filelist = []
        
        for i in range(self.nROI):
            G2filelist.append(self.FolderName + '\\' + 'ROI' + str(1 + i).zfill(4) +  'g2.dat')
            tauG2filelist.append(self.FolderName + '\\' + 'ROI' + str(1 + i).zfill(4) +  'taug2.dat')
            varG2filelist.append(self.FolderName + '\\' + 'ROI' + str(1 + i).zfill(4) +  'varg2.dat')
        

        for i in range(self.nROI):
            self.g2.append(pd.read_csv(G2filelist[i], sep='\t', engine = 'python', header = None, squeeze = True))
            self.taug2.append(pd.read_csv(tauG2filelist[i], sep='\t', engine = 'python', header = None, squeeze = True).tolist())
            try:
                self.g2var.append(pd.read_csv(varG2filelist[i], sep='\t', engine = 'python', header = None, squeeze = True))
            except FileNotFoundError:
                print('no var has been saved for ROI'+ str(1 + i).zfill(4))
        
        
        return
