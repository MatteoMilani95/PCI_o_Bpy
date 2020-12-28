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
import numpy as np
import pandas as pd
from scipy.optimize import leastsq, least_squares, curve_fit
import easygui as g
from lmfit import Minimizer, Parameters, fit_report
import os
import sys
from sys import exit


def theta1_func(H_value,R,n1,n2):
    tc=np.arcsin(n2/n1)
    H=lambda theta1 :R*np.sin(theta1)/np.cos(np.arcsin(n1/n2*np.sin(theta1))-theta1)*1/(1-np.tan(np.arcsin(n1/n2*np.sin(theta1))-theta1)/np.tan(np.arcsin(n1/n2*np.sin(theta1))))
    theta=inversefunc(H,y_values=H_value,domain=[-tc, tc])
    
    if H_value>=0:
        h=R*np.sin(theta)/np.cos(np.arcsin(n1/n2*np.sin(theta))-theta)
        theta_scattering=np.arcsin(R*np.sin(theta)/h)
    else:
        h=R*np.sin(theta)/np.cos(np.arcsin(n1/n2*np.sin(theta))-theta)
        theta_scattering=math.pi-np.arcsin(R*np.sin(theta)/h)
    
    return h,theta_scattering


def SingExp(x,variables,data=None,eps_data=None):
    """Model a decaying sine wave and subtract data."""
    amp = variables[0]
    decay = variables[1]
    baseline = variables[2]
        
    
    model = (amp * np.exp(-x/decay))**2 + baseline
    
    if data is None:
        return model
    if eps_data is None:
        return model - data
    
    return (data-model) / eps_data

def SingExp(x, amp, decay, baseline ):
    """Model a decaying sine wave and subtract data."""   
    
    model = (amp * np.exp(-x/decay))**2 + baseline
    
    return model


def DoubleExp(x, amp1, decay1, amp2, decay2, baseline ):
    """Model a decaying sine wave and subtract data."""
    
    model = (amp1 * np.exp(-x/decay1) + amp2 * np.exp(-x/decay2))**2 + baseline
     
    return model

def FromTautoTheta(tau,tau_err,T,R_particle,wavelength,nu,n):
    kb=1.38064852*10**-23
    D = kb*T/(6*math.pi*nu*(R_particle*10**-9))
    theta = 2* np.arcsin(  (1/(D*tau))**0.5*wavelength/(4*math.pi*n)  ) *360/(2*math.pi)
    theta_err = 2 * 1 / ( 1- ( wavelength / (4 * n * math.pi * 1 / ( D * tau )**0.5 ) )**2 )**0.5 * wavelength / (8 * n * math.pi) * 1 / D**0.5 * tau**-1.5 * tau_err *360/(2*math.pi)
    return D, theta, theta_err



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
        
  
        
    def LoadCI(self, FolderName,lagtime):
        #This method automatically call the method LoadInput_101_CalCI(), 
        #and the load the CI files for each ROI
        
        self.FolderName = FolderName
        self.lag = lagtime
        self.LoadInput_101_CalCI()
        
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
    
    


    
    
    
class CIbead(CIfile):
    def __init__(self,n1,n2,wavelength):
        super().__init__()
        self.Radius = 0
        self.indexrefbead = n1
        self.indexrefext = n2
        self.scatt_angle = []
        self.scatt_angle_exp = []
        self.Center = 0
        self.magnification = 2.15/357
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
        
    def SetQvector(self):
        for i in range(len(self.scatt_angle)):
            self.qvetcros.append(4*math.pi**np.sin(self.scatt_angle[i]/2)/self.wavelength) 
        return
        
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
            inner_h,scattering_angle=theta1_func(self.ROI_x_pos[i],Radius,self.indexrefbead,self.indexrefext)
            h.append(inner_h)
            self.scatt_angle.append(scattering_angle*360/(2*math.pi))
                
        return self.scatt_angle
    
    
        
      
        
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
        self.decaytime = []
        self.decaytimeerr = []
        
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
    
    def G2Normalize(self):
        for i in range(self.nROI):
            for j in range(len(self.g2[i])):
                self.g2[i][j] = self.g2[i][j]/self.g2[i][2]
        return
    
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
    
    
    def FindDecaytime(self,func,variables,plot):
        
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
            self.decaytime.append(outparam[i][0][1])
            self.decaytimeerr.append(2*np.sqrt(outparam[i][1][1][1]))
        
        goodness_fit = [] 
        for i in range(self.nROI):
            goodness_fit.append( np.sum( ( ( np.asarray(fitted_curve[i])-np.asarray(self.g2[i]))  )**2 / np.asarray(fitted_curve[i]) ) )
    
        if plot == True:
            for i in range(self.nROI):
                plt.figure() 
                plt.xscale('log')
                plt.errorbar(self.tau,self.g2[i],yerr=self.g2var[i],fmt='o',label='chi = ' + str(goodness_fit[i]))
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
    

    def FitDoubleDecaytime(self,func,variables,plot):
            
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
                self.decaytime.append(outparam[i][0][1])
                self.decaytimeerr.append(2*np.sqrt(outparam[i][1][1][1]))
            
            goodness_fit = [] 
            for i in range(self.nROI):
                goodness_fit.append( np.sum( ( ( np.asarray(fitted_curve[i])-np.asarray(self.g2[i]))  )**2 / np.asarray(fitted_curve[i]) ) )
        
            if plot == True:
                for i in range(self.nROI):
                    plt.figure() 
                    plt.xscale('log')
                    plt.errorbar(self.tau,self.g2[i],yerr=self.g2var[i],fmt='o',label='chi = ' + str(goodness_fit[i]))
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
        
        
# THE PROGRAM STRST HERE, DELETE IT WHEN THIS FILE WILL BECAME A PYTHONPACKAGE




    
'''
ciBEAD = CIbead(1.36,1.0,532*10**-9) 
ciBEAD.SetROIlist([125,370],[1780,300],[178,300])
ciBEAD.UploadInput_101_CalCI() 
'''
'''
ciBEAD = CIbead(1.36,1.0,532*10**-9) 
ciBEAD.LoadCI('E:\\Matteo\\PHD\\light_scattering\\20201211_1621_PL105D4.2E-5_896x184px_19000fs_50us_2000mW_bead_1\\out',1/19000)
g2BEAD = G2(ciBEAD.FolderName,ciBEAD.CI,ciBEAD.nROI,ciBEAD.tau)
g2BEAD.G2Calculatino()
v=np.array([0.1,0.0001,0.0])
g2BEAD.G2CutBaseLine(9)
g2BEAD.FindDecaytime(SingExp,v,plot=False)
R = 1.52
sc_ang_the_BEAD = ciBEAD.SetThetaScatt(R)
D,sc_ang_exp_BEAD,sc_ang_exp_BEAD_err =FromTautoTheta(np.asarray(g2BEAD.decaytime),g2BEAD.decaytimeerr,293.15,52.5,ciBEAD.wavelength,0.0022,ciBEAD.indexrefbead)
posBEAD = ciBEAD.ROI_x_pos


plt.figure() 
plt.errorbar(ciBEAD.ROI_x_pos,sc_ang_exp_BEAD,yerr=sc_ang_exp_BEAD_err,fmt='.',label='T=20 [C] v=0.0022 [Pa s]')
plt.plot(ciBEAD.ROI_x_pos,sc_ang_the_BEAD)
plt.xlabel('x [mm]')
plt.ylabel('scattering angle [grad]')
plt.title('R = '+ str(R)+' mm')
plt.legend(loc='lower left')
plt.savefig(g2BEAD.FolderName + '\\fit_graphs\\thetaplot'+'.png')

diff=sc_ang_exp_BEAD-sc_ang_the_BEAD
plt.figure() 
#plt.errorbar(ciBEAD.ROI_x_pos,sc_ang_exp_BEAD,yerr=sc_ang_exp_BEAD_err,fmt='.',label='T=20 [C] v=0.0026 [Pa s]')
plt.plot(ciBEAD.ROI_x_pos[0:-3],diff[0:-3],'o')
plt.xlabel('x [mm]')
plt.ylabel('difference scattering angle [grad]')
plt.title('R = '+ str(R)+' mm')
plt.legend(loc='lower left')
plt.savefig(g2BEAD.FolderName + '\\fit_graphs\\diffthetaplot'+'.png')
'''
'''
ciBEAD2 = CIbead(1.36,1.0,532*10**-9) 
ciBEAD2.LoadCI('E:\\Matteo\\PHD\\light_scattering\\20201211_1537_PL105D4.2E-5_896x184px_19000fs_50us_2000mW_bead_1\\out',1/19000)
g2BEAD2 = G2(ciBEAD2.FolderName,ciBEAD2.CI,ciBEAD2.nROI,ciBEAD2.tau)
g2BEAD2.G2Calculatino()
v=np.array([0.1,0.0001,0.0])
g2BEAD2.G2CutBaseLine(9)
g2BEAD2.FindDecaytime(SingExp,v,plot=False)
R = 1.56
sc_ang_the_BEAD2 = ciBEAD2.SetThetaScatt(R)
D2,sc_ang_exp_BEAD2,sc_ang_exp_BEAD_err2 =FromTautoTheta(np.asarray(g2BEAD2.decaytime),g2BEAD2.decaytimeerr,293.15,52.5,ciBEAD2.wavelength,0.0022,ciBEAD2.indexrefbead)
posBEAD2 = ciBEAD2.ROI_x_pos

plt.figure() 
plt.errorbar(ciBEAD.ROI_x_pos,sc_ang_exp_BEAD,yerr=sc_ang_exp_BEAD_err,fmt='.',label='t = 0 min')
plt.errorbar(ciBEAD2.ROI_x_pos,sc_ang_exp_BEAD2,yerr=sc_ang_exp_BEAD_err2,fmt='.',label='t = 44 min')
plt.plot(ciBEAD.ROI_x_pos,sc_ang_the_BEAD,color = 'tab:blue')
plt.plot(ciBEAD2.ROI_x_pos,sc_ang_the_BEAD2,color = 'tab:orange')
plt.xlabel('x [mm]')
plt.ylabel('scattering angle [grad]')
#plt.title('R = '+ str(R)+' mm')
plt.legend(loc='lower left')
plt.savefig(g2BEAD.FolderName + '\\fit_graphs\\thetaplot_comparison'+'.png')
'''

ciGel = CIbead(1.33,1.0,532*10**-9) 
ciGel.LoadCI('E:\\Matteo\\PHD\\light_scattering\\20201029_gelation2_silicagel_0036M_15VF_15ul\\out\\all_data_10_ROI',0.1)
g2Gel = G2(ciGel.FolderName,ciGel.CI,ciGel.nROI,ciGel.tau)
g2Gel.G2Calculatino()
v=np.array([0.1,0.1,100,1,0.0])
g2Gel.G2CutBaseLine(9)
g2Gel.FitDoubleDecaytime(DoubleExp,v,plot=True)
R = 1.56

'''
sc_ang_the_BEAD2 = ciBEAD2.SetThetaScatt(R)
D2,sc_ang_exp_BEAD2,sc_ang_exp_BEAD_err2 =FromTautoTheta(np.asarray(g2BEAD2.decaytime),g2BEAD2.decaytimeerr,293.15,52.5,ciBEAD2.wavelength,0.0022,ciBEAD2.indexrefbead)
posBEAD2 = ciBEAD2.ROI_x_pos

plt.figure() 
plt.errorbar(ciBEAD.ROI_x_pos,sc_ang_exp_BEAD,yerr=sc_ang_exp_BEAD_err,fmt='.',label='t = 0 min')
plt.errorbar(ciBEAD2.ROI_x_pos,sc_ang_exp_BEAD2,yerr=sc_ang_exp_BEAD_err2,fmt='.',label='t = 44 min')
plt.plot(ciBEAD.ROI_x_pos,sc_ang_the_BEAD,color = 'tab:blue')
plt.plot(ciBEAD2.ROI_x_pos,sc_ang_the_BEAD2,color = 'tab:orange')
plt.xlabel('x [mm]')
plt.ylabel('scattering angle [grad]')
#plt.title('R = '+ str(R)+' mm')
plt.legend(loc='lower left')
plt.savefig(g2BEAD.FolderName + '\\fit_graphs\\thetaplot_comparison'+'.png')
'''


