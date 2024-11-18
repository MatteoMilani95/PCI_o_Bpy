# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:23:33 2024

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
import h5py, hdf5plugin


class XRAY_DATA():
    
    def __init__(self):
        
        self.dim =0
        
        """
        processed Parameters
        
        
        """
        
        self.q = []
        self.I_q = []
        self.trm =[]
        self.epoch = []
        self.volume = []
        
        """
        raw Parameters
        
        
        """
        
        
        self.images = []
        
        
        
        self.df  = []
        self.phi = []
        self.radius = []
        self.l_c = []
        self.alpha = []
        self.clustersize = []
        
        self.ampiezza_df =[]
        self.I_q_multiplied = []
        
        self.absolute_normalization = []
        
        self.I_diff = []
       
        
        
        #self.Input_101 =[]
    def __repr__(self): 
        return '<ROI: fn%s>' % (self.path)
    
    def __str__(self):
        str_res  = '\n|---------------|'
        str_res += '\n| XRAY_DATA class: '
        str_res += '\n|--------------------+--------------------|'
        return str_res
    
    
    def load_processed(self,filelist):
        
        self.q = []
        self.I_q = []
        self.trm = []
        
        self.epoch = []
        
        
        
        for i in range(len(filelist)):
            

            
            df = pd.DataFrame(np.array(h5py.File(filelist[i])['entry_0000/PyFAI/result_ave/data']))#
            q_dat = pd.DataFrame(np.array(h5py.File(filelist[i])['entry_0000/PyFAI/result_ave/q']))
            I = np.transpose(df)
            
            trm = pd.DataFrame(np.array(h5py.File(filelist[i])['entry_0000/PyFAI/MCS/interpreted/trm_counter_trm']))
            epoch = pd.DataFrame(np.array(h5py.File(filelist[i])['entry_0000/PyFAI/MCS/interpreted/epoch']))
            
            
            n =len(I.columns)
            
            for j in range(n):
            
                self.q.append(np.asarray(q_dat).flatten())
                self.I_q.append(np.asarray(I[j]).flatten())
                self.trm.append(np.asarray(trm).flatten())
                
                if j == 0 :
                    
                    self.epoch.append(np.asarray(epoch).flatten())
                                      
                else:
                    time = np.asarray(epoch) + self.epoch[-1]
                    
                    
                    self.epoch.append(np.asarray(time))
                    
            
            
        self.dim = len(self.I_q)
                    
        return
    
    
    
    
    def load_volume(self,folder_volumes):
        
        
        
        self.volume = []
        
        data = pd.read_excel(folder_volumes,header=None)
        
        volumes = np.asarray(data[8])
        
        volumes_sorted = np.sort(volumes)
        
        volumes_sorted  = volumes_sorted [::-1]
        
        for i in range(len(volumes_sorted)):
            self.volume.append(volumes_sorted[i])
        
        return
    
    def set_phi(self,phi_i):
        
        self.phi = []
        
        if self.dim ==0:
            print('Warning: first load I_q with load_processed')
            return
        else:
            if self.dim == len(self.volume):
                for i in range(self.dim):
                    phi = phi_i *  self.volume[0] / self.volume[i]
                    self.phi.append(phi)
            else:
                print('stronzo')
                print(self.dim-len(self.volume))
                
                
                
                for i in range(len(self.volume)):
                    phi = phi_i *  self.volume[0] / self.volume[i]
                    self.phi.append(phi)
                    
                for i in range(self.dim-len(self.volume)):
                    #phi = self.phi[-1]
                    #self.phi.append(phi)
                    self.I_q.pop(-1)
                    self.q.pop(-1)
                self.dim = len(self.volume)
                
        
        
        return
    
    
    def normalize_by_large_q(self,q_start,q_stop,plot=True):
        
        
        
        
        self.I_q_normalized_high_q = [] 
        self.alpha = []
       
        start = []
        stop = []
        
        for i in range(self.dim):
            start.append(sf.find_nearest(self.q[i], q_start))
            stop.append(sf.find_nearest(self.q[i], q_stop))
        
            
        self.absolute_normalization = []
        for i in range(len(self.q)):
            self.absolute_normalization.append(np.mean(self.I_q[0][start[i]:stop[i]]))
            
           
        
        for i in range(self.dim):
            
            
            self.alpha.append(np.mean(self.I_q[0][start[i]:stop[i]])/np.mean(self.I_q[i][start[i]:stop[i]]))
            self.I_q_normalized_high_q.append(self.I_q[i]*np.mean(self.I_q[0][start[i]:stop[i]])/np.mean(self.I_q[i][start[i]:stop[i]]))
            
        if plot == True:
                
            n = len(self.q)
            colors = pl.cm.winter_r(np.linspace(0,1,n))
                
            plt.figure()
            ax = plt.axes()
    
            for i in range(self.dim):
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.plot(self.q[i][start[i]:stop[i]],self.I_q_normalized_high_q[i][start[i]:stop[i]],color=colors[i],label=r'$\varphi$ = '+str(np.round(self.phi[i],3)))
                ax.set_xlabel(r'q [$nm^{-1}$]',fontsize=15)
                ax.set_ylabel(r'I(q)/$\varphi$ ',fontsize=15)
                #ax.legend(loc='upper right')
                
                
            n = len(self.q)
            colors = pl.cm.cool_r(np.linspace(0,1,n))
                
            plt.figure()
            ax = plt.axes()
    
            for i in range(self.dim):
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.plot(self.q[i],self.I_q_normalized_high_q[i],color=colors[i],label=r'$\varphi$ = '+str(np.round(self.phi[i],3)))
                ax.set_xlabel(r'q [$nm^{-1}$]',fontsize=15)
                ax.set_ylabel(r'I(q)/$\varphi$ ',fontsize=15)
                #ax.legend(loc='upper right')

            
        
        return
    
    def index_of_phi(self,phi):
        index_phi = sf.find_nearest(self.phi, phi)
        return index_phi
    
    def reset_fit_df(self):
        
        self.df  = []
        self.ampiezza_df =[]
        self.I_q_multiplied = []
        return
    
    
    def reset_fit_cluster(self):
        
        self.df_neg  = []
        self.ampiezza_df_neg =[]
        self.I_q_multiplied_neg = []
        return
    
    
    
    
    
    def fit_fractal_dimension(self,q_start,q_stop,plot=True,phi_min=0,phi_max=1):
        
        
        start = []
        stop = []
        
        for i in range(self.dim):
            start.append(sf.find_nearest(self.q[i], q_start))
            stop.append(sf.find_nearest(self.q[i], q_stop))
            

        colors = pl.cm.cool_r(np.linspace(0,1,self.dim))
        
        for i in range(len(self.df),self.dim):
            
            if self.phi[i] >= phi_min and self.phi[i] < phi_max :
                
            
                par = []
                
                par.append(curve_fit(sf.power_law,  self.q[i][start[i]:stop[i]],self.I_q[i][start[i]:stop[i]]))
                new_I_q = par[0][0][1]*np.asarray(self.q[i])**par[0][0][0]
                df_10 = par[0][0][0]
                self.df.append(-df_10)
                self.ampiezza_df.append(par[0][0][1])
                self.I_q_multiplied.append(self.I_q[i]* self.q[i]**self.df[i])
                    
                if plot == True:
                        
                
                    plt.figure()
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
        
                            
                    plt.figure()
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
   
            else:
                return
                
                
            
            
                        
                        
        return
    
    
    def fit_to_find_cluster(self,q_start,q_stop,plot=True,phi_min=0,phi_max=1):
        
        start = []
        stop = []
        
        for i in range(self.dim):
            start.append(sf.find_nearest(self.q[i], q_start))
            stop.append(sf.find_nearest(self.q[i], q_stop))
            
        colors = pl.cm.cool_r(np.linspace(0,1,self.dim))
        
        
        for i in range(len(self.df_neg),self.dim):
            
            
            if self.phi[i] >= phi_min and self.phi[i] < phi_max :
        
                par = []
                
                
                par.append(curve_fit(sf.power_law,  self.q[i][start[i]:stop[i]],self.I_q_multiplied[i][start[i]:stop[i]]))
                    
                new_I_q = par[0][0][1]*np.asarray(self.q[i])**par[0][0][0]
                df_10 = par[0][0][0]
                self.df_neg.append(-df_10)
                self.ampiezza_df_neg.append(par[0][0][1])
                self.I_q_multiplied_neg.append(new_I_q* self.q[i]**self.df[i])
                    
                if plot == True:
                
                    plt.figure()
                    ax = plt.axes()
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                        
                    ax.plot(self.q[i],self.I_q_multiplied[i],color=colors[i],label='$\phi =$ '+str(self.phi[i]*100)+'%')
                    ax.plot(self.q[i][start[i]:stop[i]],self.I_q_multiplied[i][start[i]:stop[i]],marker='x',color='black',linestyle='',label='fitted points')
                    ax.plot(np.asarray(self.q[i]),new_I_q,color='red',label='fit')
                    ax.set_xlabel(r'q [$nm^{-1}$]',fontsize=15)
                    ax.set_ylabel(r'I(q) ',fontsize=15)
                    ax.legend()
                    
            
                        
        return
    
    
    def intersection_cluster(self,plot=True,step_plot=1):
        

        self.clustersize = []
        n = len(self.q)
        colors = pl.cm.cool_r(np.linspace(0,1,n))
        
        if len(self.df) == 0 or len(self.df_neg) == 0:
            print('fit df and df_neg before')
            return
        
        for i in range(self.dim):
            
            
            interp_func1 = interp1d(self.q[i], self.ampiezza_df[i]*np.asarray(self.q[i])**(-self.df[i]), kind='linear', fill_value="extrapolate")
            interp_func2 = interp1d(self.q[i], self.ampiezza_df_neg[i]*(np.asarray(self.q[i]))**(-self.df_neg[i])*np.asarray(self.q[i])**(-self.df[i]), kind='linear', fill_value="extrapolate")
                            
                            # Define the function to find the root of
            def find_intersection(x):
                return interp_func1(x) - interp_func2(x)
                            
                            # Use fsolve to find the intersection point
            initial_guess = 1e-2
            intersection_x = fsolve(find_intersection, initial_guess)[0]
            intersection_y = interp_func1(intersection_x)
            
            
            self.clustersize.append(2*np.pi/intersection_x)
                            
            if i % step_plot == 0:
                            
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
    
    
    def determine_ls(self,q_for_threshold,plot=True,step_plot=1):
        
        self.I_diff = []
        self.l_c = []
        
        
        for i in range(self.dim):
            self.I_diff.append(self.I_q_normalized_high_q[0][5:]-self.I_q_normalized_high_q[i][5:])
        
        index_threshold = sf.find_nearest(self.q[0], 0.7)


        threshold = self.I_q_normalized_high_q[0][index_threshold]
        zeros_difference = []
        for i in range(self.dim):
            zeros_difference.append(sf.first_below_threshold_index(self.I_diff[i], threshold))
            
        self.l_c.append(np.nan)
        
        for i in range(self.dim-1):
            self.l_c.append(2 * np.pi*1/self.q[i+1][5:][zeros_difference[i+1]])
         
            
        colors = pl.cm.coolwarm(np.linspace(0,1,self.dim))
        
        plt.figure()
        ax = plt.axes()

        ax.set_xscale('log')
        ax.set_yscale('log')
        for i in range(self.dim):
            ax.plot(self.phi[i],self.l_c[i],color=colors[i],marker='o',markersize=8,markeredgecolor='black')
            
        ax.set_xlabel(r'$\varphi$  [-]',fontsize=20)
        ax.set_ylabel(r'$l_{\rm{C}}$ [nm]',fontsize=20)
        
        plt.axhline(y = 6, color = 'r', linestyle = '-')  
        
        
            
        if plot == True:
            
            plt.figure()
            ax = plt.axes()
                    
            ax.set_xscale('log')
            ax.set_yscale('log')
                    
            for i in range(self.dim):
                     
                if i % step_plot == 0:
                    ax.plot(self.q[i][5:],self.I_diff[i],color=colors[i],linewidth=3)
                    ax.plot(self.q[i][5:][zeros_difference[i]],self.I_diff[i][zeros_difference[i]],color=colors[i],marker='o',markeredgecolor='black')
                    ax.set_xlim([7*1e-3,1.3])
                    ax.set_ylim([-100000,1e6])
                    ax.set_xlabel(r'q [$nm^{-1}$]',fontsize=15)
                    ax.set_ylabel(r'$\tilde{I}_0(q)-\tilde{I}(q)$ ',fontsize=15)
            
        return
    
    
    
    
    


def list_h5_files(folder_path, option='all'):
    """
    List files in the specified folder that end with '_ave.h5'.
    
    Parameters:
    folder_path (str): The path to the folder to search in.
    option (str): The option for listing files ('all', 'even', 'odd').
    
    Returns:
    list: A list of filenames ending with '_ave.h5' based on the specified option.
    """
    try:
        # List all files in the folder
        all_files = os.listdir(folder_path)
        # Filter files that end with '_ave.h5'
        h5_files = [os.path.join(folder_path, f) for f in all_files if f.endswith('_ave.h5')]

        # Apply the selected option
        if option == 'even':
            h5_files = [os.path.join(folder_path, f) for idx, f in enumerate(h5_files) if idx % 2 == 0]
        elif option == 'odd':
            h5_files = [os.path.join(folder_path, f) for idx, f in enumerate(h5_files) if idx % 2 != 0]
        
        return h5_files
    except FileNotFoundError:
        print("The specified folder does not exist.")
        return []



'''
def load_and_save_images(folder_path):
        
        filelist = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if 'cam' in f ]
        
        folderout = folder_path + r'/extracted_images'
        
        try:
            os.mkdir(folderout)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        
        
        for i in range(len(filelist)):
            
            df = np.array(h5py.File(filelist[i])['entry_0000/ESRF-ID02/cam/data'])
            
            n = len (df)
            
            for j in range(n):
                
                plt.imsave(folderout+'/image_dataset_'+ str( i+1 ).zfill(3)+  '_number_' + str( j+1 ).zfill(3)+'.png', df[j,410:560,420:900],cmap='gray')
        
        
        
        return
    
'''    
def load_and_save_images(folder_path, option='all'):
    """
    Load images from HDF5 files in the specified folder and save them as PNG files.
    
    Parameters:
    folder_path (str): The path to the folder containing HDF5 files.
    option (str): The option for selecting files ('all', 'even', 'odd').
    
    Returns:
    None
    """
    
    # Create a list of file paths for files containing 'cam' in their name
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if 'cam' in f]
    
    # Apply the selected option
    if option == 'even':
        filelist = [f for idx, f in enumerate(all_files) if idx % 2 == 0]
        
        folderout = os.path.join(folder_path, 'extracted_images_even')
        print(len(filelist))
        
    elif option == 'odd':
        filelist = [f for idx, f in enumerate(all_files) if idx % 2 != 0]
        folderout = os.path.join(folder_path, 'extracted_images_odd')
        print(len(filelist))
    else:  # 'all'
        filelist = all_files
    
        folderout = os.path.join(folder_path, 'extracted_images')
    
    try:
        os.mkdir(folderout)
    except FileExistsError:
        print('Directory already exists, graphs will be uploaded')
    
    for i, file in enumerate(filelist):
        with h5py.File(file, 'r') as h5file:
            df = np.array(h5file['entry_0000/ESRF-ID02/cam/data'])
        
        n = len(df)
        
    
        for j in range(n):
            plt.imsave(folderout+'/image_dataset_'+ str( i+1 ).zfill(3)+  '_number_' + str( j+1 ).zfill(3)+'.png', df[j,410:560,420:900],cmap='gray')

    


'''
    
    def load_images(self,filelist):
        
        for i in range(len(filelist)):
            
            df = np.array(h5py.File(filelist[i])['entry_0000/ESRF-ID02/cam/data'])
            
            n = len (df)
            
            for j in range(n):
                
                self.images.append(df[j,410:560,420:900])
                
            
        
        return
    
def list_cam_files(folder_path):
    """
    List all files in the specified folder that contain 'cam' in the name 
    and end with '_ave.h5'.
        
    Parameters:
    folder_path (str): The path to the folder to search in.
        
    Returns:
    list: A list of filenames that contain 'cam' and end with '_ave.h5'.
    """
    h5_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if 'cam' in f ]
    
    outfold = folder_path + r'/out'
    
    try:
        os.mkdir(outfold)
    except FileExistsError:
        print('directory already existing, graphs will be uploaded')

'''

