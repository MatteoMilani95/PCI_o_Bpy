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
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.optimize import fsolve



class XRAY_DATA():
    
    def __init__(self):
        """Initialize ROIs
        
        Parameters
        ----------
        Filename: complete filename of the CI file
        
        
        """
        
        self.q = []
        self.I_q = []
        self.df  = []
        self.phi = []
        self.radius = []
        self.lc = []
        self.alpha = []
        self.volume = []
        self.absolute_normalization = []
       
        
        
        #self.Input_101 =[]
    def __repr__(self): 
        return '<ROI: fn%s>' % (self.path)
    
    def __str__(self):
        str_res  = '\n|---------------|'
        str_res += '\n| XRAY_DATA class: '
        str_res += '\n|--------------------+--------------------|'
        return str_res
    

    
    def load_Soleil(self,path,phi):
        
        q_data = np.loadtxt(path,skiprows=25,usecols=(0))
        I_q_data = np.loadtxt(path,skiprows=25,usecols=(1))
        
        self.q.append(q_data*10)
        self.I_q.append(I_q_data)
        self.phi.append(phi)
        
        
        return
    
    def load_Averaged_back_Soleil(self,path):
        
        self.q_back = np.loadtxt(path,skiprows=25,usecols=(0))
        self.I_q_back = np.loadtxt(path,skiprows=25,usecols=(1))
        
        self.I_q_non_back = []
        
        
        for i in range(len(self.q)):
            self.I_q_non_back.append(self.I_q[i])
            self.I_q[i] = self.I_q[i] - self.I_q_back
    
   
    
    def load_back_Soleil(self,path):
        
        self.q_back = np.loadtxt(path,skiprows=45,usecols=(0))
        self.I_q_back = np.loadtxt(path,skiprows=45,usecols=(1))
        
        self.I_q_non_back = []
        
        
        for i in range(len(self.q)):
            self.I_q_non_back.append(self.I_q[i])
            self.I_q[i] = self.I_q[i] - self.I_q_back

        
        return
    
    def load_averaged_back_Soleil(self,path):
        
        self.q_back = np.loadtxt(path,skiprows=25,usecols=(0))
        self.I_q_back = np.loadtxt(path,skiprows=25,usecols=(1))
        
        self.I_q_non_back = []
        
        
        for i in range(len(self.q)):
            self.I_q_non_back.append(self.I_q[i])
            self.I_q[i] = self.I_q[i] - self.I_q_back

        
        return
    
    def load_diluted(self,path,plot=True):
        
        path_back = r'H:\Hierarchical_compaction\Swing_march24\20231401\Swing_march24_processed_data_Xary\gels_and_suspension_in_capillaries\back\background_00856{00000}_AzInt_Px_0.dat'
        
        back_diluted = np.loadtxt(path_back,skiprows=43,usecols=(1))
        
        self.q_diluted = np.loadtxt(path,skiprows=23,usecols=(0))
        self.I_q_diluted = np.loadtxt(path,skiprows=23,usecols=(1))-back_diluted
        
        self.I_q_structure = []
        

        
        
        for i in range(len(self.q)):
            
            self.I_q_structure.append(self.I_q_normalized_high_q[i] /self.I_q_diluted)
        
        
        
        return
    
    
    def load_processed_data(self,name):
        
        path = name +'\\folder_raw_data'
        
        
        
        files = os.listdir(path)
        
       
        
        for i in range(len(files)):
        
            self.q.append(np.loadtxt( name +'\\folder_raw_data\\'+files[i],usecols=(0)))
            self.I_q.append(np.loadtxt( name +'\\folder_raw_data\\'+files[i],usecols=(1)))
            
            
        phi = np.loadtxt(name + '\\general_info.txt',usecols=(0))
        df = np.loadtxt(name + '\\general_info.txt',usecols=(1))
        lc= np.loadtxt(name + '\\general_info.txt',usecols=(2))
        alpha = np.loadtxt(name + '\\general_info.txt',usecols=(3)) 
        abs_n = np.loadtxt(name + '\\general_info.txt',usecols=(4)) 
        volume = np.loadtxt(name + '\\general_info.txt',usecols=(5))
        
        for i in range(len(files)):
            
            self.phi.append(phi[i] )
            self.df.append(df[i])
            self.lc.append( lc[i])
            self.alpha.append(alpha[i])
            self.absolute_normalization.append(abs_n[i])
            self.volume.append(volume[i] )
            
            
        
        return
    
    def load_Philippe(self,path,raw_to_skip):
        
        data = pd.read_csv(path,skiprows=range(0,raw_to_skip),delimiter=r"\s+", header=None)
        
        
        for i in range(len(data.columns)-1):
            self.I_q.append(np.asarray(data[i+1]))
            self.q.append(np.asarray(data[0]))
            
        
        return
    
    
            
    def load_gealtion_capillary_20230325(self,path,phi):
        
        path_back = r'H:\Hierarchical_compaction\Swing_march24\20231401\Swing_march24_processed_data_Xary\gels_and_suspension_in_capillaries\back\background_00856{00000}_AzInt_Px_0.dat'
        
        
        q_data = np.loadtxt(path,skiprows=20,usecols=(0))
        I_q_data = np.loadtxt(path,skiprows=20,usecols=(1))
        
        self.q.append(q_data*10)
        self.I_q.append(I_q_data)
        self.phi.append(phi)
        
        
        self.q_back = np.loadtxt(path_back,skiprows=40,usecols=(0))
        self.I_q_back = np.loadtxt(path_back,skiprows=40,usecols=(1))
        
        self.I_q_non_back = []
        
        
        for i in range(len(self.q)):
            self.I_q_non_back.append(self.I_q[i])
            self.I_q[i] = self.I_q[i] - self.I_q_back
        
        
        
        
        return
    
    
    
    def plot_data_Xray(self,xlim,ylim):
        
        n = len(self.q)
        colors = pl.cm.cool_r(np.linspace(0,1,n))

        
        fig = plt.figure()
        ax = plt.axes()
        
        for i in range(len(self.q)):

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.plot(self.q[i],self.I_q[i],color=colors[i],label=r'$\varphi$ = '+str(np.round(self.phi[i],3)))
            ax.set_xlabel(r'q [$nm^{-1}$]',fontsize=15)
            ax.set_ylabel(r'I(q) ',fontsize=15)
            ax.legend(loc='upper right')
            ax.set_xlim([xlim[0],xlim[1]])
            ax.set_ylim([ylim[0],ylim[1]])
            

            
            
    
    
    def interpolate_phi(self,Radius_pixel,conversion,time,phi_initial,plot=True):
        
        Raius_mm     = Radius_pixel / conversion*1.5
        volume_fraction = []

        for i in range(len(Radius_pixel)):
            volume_fraction.append(phi_initial * (Raius_mm[0]/Raius_mm[i])**3 )
        
        interpolated_phi = interpolate.interp1d(time,volume_fraction)
        
        self.time = np.linspace(time[0],time[-1],len(self.q))
        
        for i in range(len(self.q)):
            
            self.phi.append(interpolated_phi(self.time[i]))
        
        if plot ==True:
            fig = plt.figure()
            ax = plt.axes()
            ax.plot(self.time,self.phi,marker='o',color='silver',markeredgecolor='dimgray')
            
        
    def fit_phi(self,Radius_pixel,conversion,time,phi_initial,plot=True):
        
        self.phi =[]
        self.time =[]
        self.V_0 = []
        self.Delta_V = []
        
        Raius_mm     = Radius_pixel / conversion*1.5

        #for i in range(len(Radius_pixel)):
        #    volume_fraction.append(phi_initial * (Raius_mm[0]/Raius_mm[i])**3 )
        
        
        self.time = np.linspace(time[0],time[-1],len(self.q))
        
        par = []

        par.append(curve_fit(sf.line,  time,Raius_mm,bounds=([-np.inf,Raius_mm[0]-0.01],[np.inf,Raius_mm[0]+0.01])))
        fitted_radius = sf.line(self.time,par[0][0][0],par[0][0][1])
        
        for i in range(len(self.q)):
            self.radius.append( fitted_radius[i])
        
        
        
        
        
        self.V_0.append( 4/3 * np.pi * Raius_mm[0]**3  )
        
        for i in range(len(fitted_radius)):
            
            
            
            self.Delta_V.append(self.V_0[0] - 4/3 * np.pi * fitted_radius[i]**3 )
        
        
        
        
        data_phi=[]
        for i in range(len(Radius_pixel)):
            data_phi.append(phi_initial * (Raius_mm[0]/Raius_mm[i])**3 )
        
        for i in range(len(self.q)):
            
            
            self.phi.append(phi_initial *(Raius_mm[0]/np.asarray(sf.line(self.time[i],par[0][0][0],par[0][0][1])))**3)
        
        if plot ==True:
            
            fig = plt.figure()
            ax = plt.axes()
            ax.plot(self.time,fitted_radius,color='red',label='fit')
            ax.plot(time,Raius_mm,linestyle='',marker='o',color='silver',markeredgecolor='dimgray',label='data')
            ax.set_xlabel(r't [min]',fontsize=15)
            ax.set_ylabel(r'R [mm] ',fontsize=15)
            ax.legend()
            
            
            fig = plt.figure()
            ax = plt.axes()
            
            ax.plot(self.time,self.Delta_V,linestyle='',marker='o',color='silver',markeredgecolor='dimgray',label='data')
            ax.set_xlabel(r't [min]',fontsize=15)
            ax.set_ylabel(r'$\Delta V$ [mm] ',fontsize=15)
            ax.legend()
            
            fig = plt.figure()
            ax = plt.axes()
            ax.plot(self.time,self.phi,color='red',label='fit')
            ax.plot(time,data_phi,linestyle='',marker='o',color='silver',markeredgecolor='dimgray',label='data')
            ax.set_xlabel(r't [min]',fontsize=15)
            ax.set_ylabel(r'$\varphi$ [mm] ',fontsize=15)
            ax.legend()
      
                
        return
    
    def find_I_restructured(self):
        
        self.I_restructured = []
        
        self.I_difference = []

        for i in range(len(self.I_q_normalized_high_q)):
        
            self.I_restructured.append( self.V_0 / self.Delta_V[i] * ( self.I_q_normalized_high_q[i] - (1 -  self.Delta_V[i] / self.V_0) *  self.I_q_normalized_high_q[0] ))  
        
        for i in range(len(self.I_q_normalized_high_q)):
            
            self.I_difference.append( self.I_q_normalized_high_q[0] - self.I_restructured[i])
       
        return
    
    def fit_Guinier_plateau(self,q_start,q_stop,plot=True):
        
        start = []
        stop = []
        
        for i in range(len(self.q)):
            start.append(sf.find_nearest(self.q[i], q_start[i]))
            stop.append(sf.find_nearest(self.q[i], q_stop[i]))
            
        
        self.Rg  = []
        
        n = len(self.q)
        colors = pl.cm.cool_r(np.linspace(0,1,n))
        
        for i in range(len(self.q)):
        
            
            popt,pcov = curve_fit(sf.Guinier_lin_log,  self.q[i][start[i]:stop[i]],np.log(self.I_q_normalized_high_q[i][start[i]:stop[i]]),bounds=([0,10],[10000000,10000]))
            
            self.Rg.append(popt[1])
            
                
            if plot == True:
            
                fig = plt.figure()
                ax = plt.axes()
                ax.set_xscale("log")
                ax.set_yscale("log")
                    
                ax.plot(self.q[i],self.I_q_normalized_high_q[i],color=colors[i],label='$\phi =$ '+str(self.phi[i]*100)+'%')
                ax.plot(self.q[i][start[i]:stop[i]],np.exp(sf.Guinier_lin_log(self.q[i][start[i]:stop[i]],popt[0],popt[1])),marker='x',color='black',linestyle='',label='fitted points')
                
                ax.set_xlabel(r'q [$nm^{-1}$]',fontsize=15)
                ax.set_ylabel(r'I(q) ',fontsize=15)
                ax.legend()
        
        return
    
    def fit_Fisher_Burford(self,q_start,q_stop,plot=True):
        
        
        start = []
        stop = []
        
        for i in range(len(self.q)):
            start.append(sf.find_nearest(self.q[i], q_start[i]))
            stop.append(sf.find_nearest(self.q[i], q_stop[i]))
            
        
        self.df_FB  = []
        self.clustersize_FB = []
        n = len(self.q)
        colors = pl.cm.cool_r(np.linspace(0,1,n))
        
        for i in range(len(self.q)):
        
            
            popt,pcov = curve_fit(sf.Fisher_Burford,  self.q[i][start[i]:stop[i]],np.log(self.I_q_normalized_high_q[i][start[i]:stop[i]]),bounds=([0,1,10],[10000000,3,10000]))
            
            self.df_FB.append(popt[1])
            self.clustersize_FB.append(popt[2])
            
            
                
                
            if plot == True:
            
                fig = plt.figure()
                ax = plt.axes()
                ax.set_xscale("log")
                ax.set_yscale("log")
                    
                ax.plot(self.q[i],self.I_q_normalized_high_q[i],color=colors[i],label='$\phi =$ '+str(self.phi[i]*100)+'%')
                ax.plot(self.q[i][start[i]:stop[i]],np.exp(sf.Fisher_Burford(self.q[i][start[i]:stop[i]],popt[0],popt[1],popt[2])),marker='x',color='black',linestyle='',label='fitted points')
                
                ax.set_xlabel(r'q [$nm^{-1}$]',fontsize=15)
                ax.set_ylabel(r'I(q) ',fontsize=15)
                ax.legend()
        
        
        
        return
    
    def fit_fractal_dimension(self,q_start,q_stop,plot=True):
        
        
        start = []
        stop = []
        
        for i in range(len(self.q)):
            start.append(sf.find_nearest(self.q[i], q_start[i]))
            stop.append(sf.find_nearest(self.q[i], q_stop[i]))
            
        
        self.df  = []
        self.ampiezza_df =[]
        self.I_q_multiplied = []
        n = len(self.q)
        colors = pl.cm.cool_r(np.linspace(0,1,n))
        
        for i in range(len(self.q)):
        
            par = []
            try:
                par.append(curve_fit(sf.power_law,  self.q[i][start[i]:stop[i]],self.I_q[i][start[i]:stop[i]]))
                new_I_q = par[0][0][1]*np.asarray(self.q[i])**par[0][0][0]
                df_10 = par[0][0][0]
                self.df.append(-df_10)
                self.ampiezza_df.append(par[0][0][1])
                self.I_q_multiplied.append(self.I_q[i]* self.q[i]**self.df[i])
                
                if plot == True:
            
                    fig = plt.figure()
                    ax = plt.axes()
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    
                    ax.plot(self.q[i],self.I_q[i],color=colors[i],label='$\phi =$ '+str(self.phi[i]*100)+'%')
                    ax.plot(self.q[i][start[i]:stop[i]],self.I_q[i][start[i]:stop[i]],marker='x',color='black',linestyle='',label='fitted points')
                    ax.plot(np.asarray(self.q[i]),new_I_q,color='red',label='fit')
                    ax.set_xlabel(r'q [$nm^{-1}$]',fontsize=15)
                    ax.set_ylabel(r'I(q) ',fontsize=15)
                    ax.legend()
                    
                    ax.text(0.001, 1e5, r'$d_f$ = '+str( np.round(df_10,2)), size=14,ha="left", va="top",bbox=dict(boxstyle="square",ec=(0.5, 0.5, 0.5),fc=(1., 0.8, 0.8)))
                    #plt.savefig(r'C:\Users\Matteo\Desktop\PHD\paper_2\20240611_discussion\fit'+str(np.round(self.phi[i]*100,2))+'.pdf',dpi=300,bbox_inches='tight',transparent=True)

                    
                    fig = plt.figure()
                    ax = plt.axes()
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    
                    ax.plot(self.q[i],self.I_q[i]* self.q[i]**self.df[i],color=colors[i],label='$\phi =$ '+str(self.phi[i]*100)+'%')
                    ax.plot(self.q[i][start[i]:stop[i]],self.I_q[i][start[i]:stop[i]]* self.q[i][start[i]:stop[i]]**self.df[i],marker='x',color='black',linestyle='',label='fitted points')
                    ax.plot(self.q[i],new_I_q* self.q[i]**self.df[i],color='red',label='fit')

                    ax.set_xlabel(r'q [$nm^{-1}$]',fontsize=15)
                    ax.set_ylabel(r'I(q) ',fontsize=15)
                    ax.legend()
                    
                    ax.set_xlim([8*1e-3,1])
                    #ax.set_ylim([1e-3,2])
                    #plt.savefig(r'C:\Users\Matteo\Desktop\PHD\paper_2\20240611_discussion\int'+str(np.round(self.phi[i]*100,2))+'.pdf',dpi=300,bbox_inches='tight',transparent=True)


            except ValueError: 
                self.df.append(3)
                
                
            
            
                        
                        
        return
    
    
    
    def fit_to_find_cluster(self,q_start,q_stop,plot=True):
        
        start = []
        stop = []
        
        for i in range(len(self.q)):
            start.append(sf.find_nearest(self.q[i], q_start[i]))
            stop.append(sf.find_nearest(self.q[i], q_stop[i]))
            
        
        self.df_neg  = []
        self.ampiezza_df_neg =[]
        self.I_q_multiplied_neg = []
        n = len(self.q)
        colors = pl.cm.cool_r(np.linspace(0,1,n))
        
        
        for i in range(len(self.q)):
        
            par = []
            try:
                print(i)
                par.append(curve_fit(sf.power_law,  self.q[i][start[i]:stop[i]],self.I_q_multiplied[i][start[i]:stop[i]]))
                
                new_I_q = par[0][0][1]*np.asarray(self.q[i])**par[0][0][0]
                df_10 = par[0][0][0]
                self.df_neg.append(-df_10)
                self.ampiezza_df_neg.append(par[0][0][1])
                self.I_q_multiplied_neg.append(new_I_q* self.q[i]**self.df[i])
                
                if plot == True:
            
                    fig = plt.figure()
                    ax = plt.axes()
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    
                    ax.plot(self.q[i],self.I_q_multiplied[i],color=colors[i],label='$\phi =$ '+str(self.phi[i]*100)+'%')
                    ax.plot(self.q[i][start[i]:stop[i]],self.I_q_multiplied[i][start[i]:stop[i]],marker='x',color='black',linestyle='',label='fitted points')
                    ax.plot(np.asarray(self.q[i]),new_I_q,color='red',label='fit')
                    ax.set_xlabel(r'q [$nm^{-1}$]',fontsize=15)
                    ax.set_ylabel(r'I(q) ',fontsize=15)
                    ax.legend()
                    
            except ValueError: 
                self.df.append(3)
                        
        return
    
    def intersection_cluster(self,plot=True):
        
        self.intersection_x= []
        self.intersection_y= []
        n = len(self.q)
        colors = pl.cm.cool_r(np.linspace(0,1,n))
        
        for i in range(len(self.q)):
            interp_func1 = interp1d(self.q[i], self.ampiezza_df[i]*np.asarray(self.q[i])**(-self.df[i]), kind='linear', fill_value="extrapolate")
            interp_func2 = interp1d(self.q[i], self.ampiezza_df_neg[i]*(np.asarray(self.q[i]))**(-self.df_neg[i])*np.asarray(self.q[i])**(-self.df[i]), kind='linear', fill_value="extrapolate")
                        
                        # Define the function to find the root of
            def find_intersection(x):
                return interp_func1(x) - interp_func2(x)
                        
                        # Use fsolve to find the intersection point
            initial_guess = 1e-2
            intersection_x = fsolve(find_intersection, initial_guess)[0]
            intersection_y = interp_func1(intersection_x)
                        
            self.intersection_x.append(intersection_x)
            self.intersection_y.append(intersection_y)
                        
            print(f"Intersection point: x = {intersection_x}, y = {intersection_y}")
                        
            if plot==True:            # Plot the results
                plt.figure(figsize=(8, 6))
                
                ax = plt.axes()
                ax.set_xscale('log')
                ax.set_yscale('log')
                
                
                plt.plot(self.q[i], self.I_q[i],color=colors[i],label='$\phi =$ '+str(self.phi[i]*100)+'%')
                #plt.plot(vf_05.q[i],vf_05.I_q_multiplied[i])
                x_range = np.linspace(min(self.q[i]), max(self.q[i]), 500)
                plt.plot(x_range, interp_func1(x_range), '-', label='Interpolated Power Law 1')
                plt.plot(x_range, interp_func2(x_range), '-', label='Interpolated Power Law 2')
                plt.plot(intersection_x, intersection_y, 'ro',markersize=10, label='Intersection Point')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()
           
        return
    
    def determine_particle_size(self,amplitude,particle_size,polydispersity,start,stop,plot=True):
        
        
        form_factor = []
        
        for j in range(len(self.q[0])):
            form_factor.append(sf.particle_size(self.q[0][j],amplitude,particle_size,polydispersity))

        self.form_factor_analytical = np.asarray(form_factor)
        
        
            
        fig = plt.figure()
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
            
       
        ax.plot(self.q[0],self.I_q[0]/self.phi[0],color='blue')
        ax.plot(self.q[0][start:-stop],self.I_q[0][start:-stop]/self.phi[0],marker='x',color='black',linestyle='',label='points')
        ax.plot(self.q[0],self.form_factor_analytical,color='red',label='superposed curve')
        ax.set_xlabel(r'q [$nm^{-1}$]',fontsize=15)
        ax.set_ylabel(r'I(q) ',fontsize=15)
        ax.legend()
        ax.set_xlim([0.1,2])
            
        
        
        return
    
    def normalize_by_large_q(self,q_start,q_stop,plot=True):
        
        
        
        
        self.I_q_normalized_high_q = [] 
        self.alpha = []
       
        start = []
        stop = []
        
        for i in range(len(self.q)):
            start.append(sf.find_nearest(self.q[i], q_start))
            stop.append(sf.find_nearest(self.q[i], q_stop))
        
            
        self.absolute_normalization = []
        for i in range(len(self.q)):
            self.absolute_normalization.append(np.mean(self.I_q[0][start[i]:stop[i]]))
            
           
        
        for i in range(len(self.q)):
            
            
            self.alpha.append(np.mean(self.I_q[0][start[i]:stop[i]])/np.mean(self.I_q[i][start[i]:stop[i]]))
            self.I_q_normalized_high_q.append(self.I_q[i]*np.mean(self.I_q[0][start[i]:stop[i]])/np.mean(self.I_q[i][start[i]:stop[i]]))
            
        if plot == True:
                
            n = len(self.q)
            colors = pl.cm.winter_r(np.linspace(0,1,n))
                
            fig = plt.figure()
            ax = plt.axes()
    
            for i in range(len(self.q)):
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.plot(self.q[i][start[i]:stop[i]],self.I_q_normalized_high_q[i][start[i]:stop[i]],color=colors[i],label=r'$\varphi$ = '+str(np.round(self.phi[i],3)))
                ax.set_xlabel(r'q [$nm^{-1}$]',fontsize=15)
                ax.set_ylabel(r'I(q)/$\varphi$ ',fontsize=15)
                ax.legend(loc='upper right')
                
                
            n = len(self.q)
            colors = pl.cm.cool_r(np.linspace(0,1,n))
                
            fig = plt.figure()
            ax = plt.axes()
    
            for i in range(len(self.q)):
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.plot(self.q[i],self.I_q_normalized_high_q[i],color=colors[i],label=r'$\varphi$ = '+str(np.round(self.phi[i],3)))
                ax.set_xlabel(r'q [$nm^{-1}$]',fontsize=15)
                ax.set_ylabel(r'I(q)/$\varphi$ ',fontsize=15)
                ax.legend(loc='upper right')

            
        
        return
    
   

    
    def normalize_by_phi(self,xlim,ylim,plot=True):
        
        self.I_q_normalized_phi = [] 
        
        

        for i in range(len(self.q)):
        
            self.I_q_normalized_phi.append(self.I_q[i]/self.phi[i])

        if plot == True:
            fig = plt.figure()
            ax = plt.axes()
            
            n = len(self.q)
            colors = pl.cm.cool_r(np.linspace(0,1,n))
            
            for i in range(len(self.q)):
    
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.plot(self.q[i],self.I_q_normalized_phi[i],color=colors[i],label=r'$\varphi$ = '+str(np.round(self.phi[i],3)))
                ax.set_xlabel(r'q [$nm^{-1}$]',fontsize=15)
                ax.set_ylabel(r'I(q)/$\varphi$ ',fontsize=15)
                ax.legend(loc='upper right')
                ax.set_xlim([xlim[0],xlim[1]])
                ax.set_ylim([ylim[0],ylim[1]])
            
            
        return
    
    def index_of_phi(self,phi):
        index_phi = sf.find_nearest(self.phi, phi)
        return index_phi
    
    
    def save_normalized_plot(self,out_folder):
        
        n = len(self.q)
        colors_d = pl.cm.copper_r(np.linspace(0,1,n))
        
        
        plt.figure()
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        
        for i in range(len(self.q)):
        
            ax.set_xlim([0.06,1.7])
            ax.set_ylim([0.04,300])
            
            ax.plot(self.q[i],self.I_q[i],linewidth=3,color=colors_d [i],label=r'$\varphi=$' +str(np.round(self.phi[i]*100,0))+r' $\%$')
        #ax.plot(Silicagel_20230411.q[3][start:-stop],0.045*Silicagel_20230411.form_factor_analytical,color='red',linewidth=3)
        
        ax.set_xlabel(r'q [$nm^{-1}$]',fontsize=15)
        ax.set_ylabel(r'I(q) ',fontsize=15)
        #ax.legend()
        plt.savefig(out_folder,dpi=300,bbox_inches='tight')
        

        return
    
    def set_cluster_size(self,clustersize,plot=True):
        
        self.clustersize = []
        self.index_cluster = []
        
        for i in range(len(self.q)):
            self.clustersize.append(2*np.pi/clustersize[i])
            
        if plot == True:
            
           
            
            n = len(self.q)
            colors_d = pl.cm.cool_r(np.linspace(0,1,n))
            
            for i in range(len(self.q)):
                
                index_cluster = sf.find_nearest(self.q[i], clustersize[i])
                self.index_cluster.append(index_cluster)
                
                plt.figure()
                ax = plt.axes()
                ax.set_xscale("log")
                ax.set_yscale("log")
                
                ax.plot(self.q[i],self.I_q[i],linewidth=3,color=colors_d [i],label=r'$\varphi=$' +str(np.round(self.phi[i]*100,0))+r' $\%$')
                ax.plot(self.q[i][index_cluster],self.I_q[i][index_cluster],linestyle='',marker='o',markersize=12,color='black',label=r'$\zeta=$' +str(self.clustersize[i])+r' nm')
                ax.legend()
            
            
            
            
        return
    
    
    
    def find_cluster_size(self,point,plot=True):
        
        self.clustersize = []
        self.index_cluster = []
        self.index_plateau = []
        
            
        if plot == True:
            
           
            
            n = len(self.q)
            colors_d = pl.cm.cool_r(np.linspace(0,1,n))
            
            for i in range(len(self.q)):
                
                index_max = sf.find_nearest(self.q[i], point[i])
                
                index_cluster = sf.find_nearest(self.I_q[i][index_max],self.ampiezza_df[i]*np.asarray(self.q[i])**-self.df[i] )
                self.index_cluster.append(index_cluster)
                self.clustersize.append(1/self.q[i][index_cluster])
                self.index_plateau.append(index_max)
                
                plt.figure()
                ax = plt.axes()
                ax.set_xscale("log")
                ax.set_yscale("log")
                
                ax.plot(self.q[i],self.ampiezza_df[i]*np.asarray(self.q[i])**-self.df[i], color = 'r', linestyle = '-' )
                plt.axhline(y = self.I_q[i][index_max], color = 'r', linestyle = '-')
                ax.plot(self.q[i],self.I_q[i],linewidth=3,color=colors_d [i],label=r'$\varphi=$' +str(np.round(self.phi[i]*100,0))+r' $\%$')
                ax.plot(self.q[i][index_cluster],self.I_q[i][index_cluster],linestyle='',marker='o',markersize=12,color='black',label=r'$\zeta=$' +str(self.clustersize[i])+r' nm')
                ax.legend()
            
            
            
            
        return
    
    def append_volume(self,volume):
        
        self.volume = volume
        
        return
    

    
    def append_lc(self,lc):
        
        self.lc = lc
        
        return
    
    def save_results(self,name):
        
        self.outfold = name
        
        try:
            os.mkdir(self.outfold)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        fname = name+'\\general_info.txt'
        np.savetxt(fname,np.c_[self.phi,self.df,self.lc,self.alpha,self.absolute_normalization,self.volume])
        
        try:
            os.mkdir(self.outfold+'\\folder_raw_data')
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
            
        int_phi = []
            
        for i in range(len(self.phi)):
            
            int_phi.append(self.phi[i]*10000)
            
            
        bar = np.rint(int_phi).astype(int)
        
            
        for i in range(len(self.phi)):
            
            name_raw = self.outfold+'\\folder_raw_data\\vf_' + str( i ).zfill(5) +'_q_Iqnormalized_Iqraw.txt'
            
            np.savetxt(name_raw,np.c_[self.q[i],self.I_q_normalized_high_q[i],self.I_q[i]])
        
        
        return
    
    def pick_from_other_object(self,q,I_q,phi):
        
        self.phi.append(phi)
        self.q.append(q)
        self.I_q.append(I_q)
        
        return
    
    
    
    
    
    
   