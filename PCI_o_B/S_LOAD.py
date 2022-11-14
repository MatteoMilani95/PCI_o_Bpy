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



class SINGLE_LOAD():
    
    def __init__(self):
        """Initialize ROIs
        
        Parameters
        ----------
        Filename: complete filename of the CI file
        
        
        """
        
        self.F_N = []
        self.time  = []
        self.d  = []
        self.phi = []
        
        self.d_0 = 0
        
        self.input_folder = []
        
        
        
        #self.Input_101 =[]
    def __repr__(self): 
        return '<ROI: fn%s>' % (self.path)
    
    def __str__(self):
        str_res  = '\n|---------------|'
        str_res += '\n| SINGLE_LOAD class: '
        str_res += '\n|--------------------+--------------------|'
        str_res += '\n| objects: '
        str_res += '\n|--------------------+--------------------|'
        str_res += '\n| Normal Force, F_N   : '  + str(len(self.F_N))
        str_res += '\n| time, t             : '  + str(len(self.time))
        str_res += '\n| distance, d         : '  + str(len(self.d))
        str_res += '\n| contact point, d_0  : '  + str(self.d_0)

        str_res += '\n|--------------------+--------------------|'
        str_res += '\n| methods: '
        str_res += '\n|--------------------+--------------------|'
        str_res += '\n| load_data_from_rhometer : load from csv F_N, time and d'
        str_res += '\n| set_d_0                 : set d_0'
        str_res += '\n|--------------------+--------------------|'
        return str_res
    
    def fodel_for_savings(self,name):
        
        self.outfold = name
        
        try:
            os.mkdir(self.outfold)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        return
    
    def save_results(self):
        
        try:
            os.mkdir(self.outfold)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
            
            
        dict = {'phi': self.phi, 'contact points': self.d_0,'Youngs modulus':self.Young_modulus,'input folder':self.input_folder}
        
        df = pd.DataFrame(dict) 
    
        # saving the dataframe 
        df.to_csv(self.outfold + '\\general_informations.csv') 
        
        try:
            os.mkdir(self.outfold + '\\cvs_results_F_N_vs_d0_d')
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        for i in range(len(self.d)):
            dict = {'d0 - d': self.d_0[i] - self.d[i], 'Normal Force': self.F_N[i]}
            df = pd.DataFrame(dict) 
    
            # saving the dataframe 
            df.to_csv(self.outfold + '\\cvs_results_F_N_vs_d0_d' + '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent.csv')
            
        
        
        return


    def load_data_from_rhometer(self,path,phi,skip=[]):
        a = []
        
        self.input_folder.append(path)
        
        #n_raw = pd.read_csv(path, index_col=None,skiprows=5,nrows=1,usecols=[2],sep='\t', decimal=",",encoding='UTF-16 LE')
     
        if len(skip) ==0:
            a.append(pd.read_csv(path, index_col=None,skiprows=10,names= ['Normal Force','time','Distance'],usecols=[2,4,5],sep='\t', decimal=",",encoding='UTF-16 LE'))
        
        else:
            
            a.append(pd.read_csv(path, index_col=None,skiprows=[0,1,2,3,4,5,6,7,8,9,10]+skip,names= ['Normal Force','time','Distance'],usecols=[2,4,5],sep='\t', decimal=",",encoding='UTF-16 LE'))

        
        self.F_N.append( np.asarray(a[0]['Normal Force']))
        self.time.append(np.asarray(a[0]['time']))
        self.d.append(np.asarray(a[0]['Distance']))
        
        self.d_0 = []
        
        self.phi.append(phi)
        
        return 
    
    def set_d_0(self,d_0):
        if len(d_0) != len(self.d):
            print('worning: number of d_0 different from the number of experiment')
            print('no contact has benn set')
            return
        
        self.d_0 = []
        
        for i in range(len(self.d)):
            self.d_0.append(d_0[i])
        
        return
    
    def set_baseline_noise(self,baseline):
        
        for i in range(len(self.d)):
            self.F_N[i] = self.F_N[i] - baseline[i] 
        
        
        return
    
    
    def cut_tails(self,npoints,plot=False):
        
        if len(npoints) != len(self.d):
            print('worning: list of points to cut has different lenght from the number of experiment')
            print('no operation has been done')
            return
        
        lst = []
        
        for i in range(len(self.d)):
            lst.append(list(range(len(self.d[i])-npoints[i],len(self.d[i]))))
        
            if plot == True:
                plt.figure()
                ax = plt.axes()
                ax.semilogx(self.d[i],self.F_N[i],marker='.',linestyle='')
                ax.semilogx(self.d[i][lst[i]],self.F_N[i][lst[i]],marker='.',linestyle='',color='red',label='phi = '+ str(self.phi[i]) + ', delete '+str(npoints[i])+' points')
                ax.set_xlabel(r'$d$' + str(' ') + ' [mm]',fontsize=18)
                ax.invert_xaxis()
                ax.set_ylabel(r'$F_N$' + str(' ') + ' [N]',fontsize=18)
                ax.legend(loc='upper right')
                
            self.d[i] = np.delete(self.d[i],lst[i],0)
            self.F_N[i] =    np.delete(self.F_N[i],lst[i],0)
            
            
    
        return 
    
    def cut_begin(self,plot=False):
        
        lst = []
        
        for i in range(len(self.d)):
            lst.append([i for i,v in enumerate(self.d_0[i] - self.d[i]) if v < 0])
        
       
        
       
        
            if plot == True:
                plt.figure()
                ax = plt.axes()
                ax.semilogx(self.d[i],self.F_N[i],marker='.',linestyle='')
                ax.semilogx(self.d[i][lst[i]],self.F_N[i][lst[i]],marker='.',linestyle='',color='red',label='phi = '+ str(self.phi[i])+ ', delete '+str(len(lst[i]))+' points')
                ax.set_xlabel(r'$d$' + str(' ') + ' [mm]',fontsize=18)
                ax.set_ylabel(r'$F_N$' + str(' ') + ' [N]',fontsize=18)
                ax.invert_xaxis()
                #ax.legend(loc='upper right')
                
            self.d[i] = np.delete(self.d[i],lst[i],0)
            self.F_N[i] =    np.delete(self.F_N[i],lst[i],0)
            
            
    
        return 
    
    
    
    def Fit_Hertz_Model(self,nu,lim_max,plot=True):
        
        self.Young_modulus = []
        base = []
        for i in range(len(self.d)):
            Force_Hertz = lambda d , E , C:  4/3 * ( E * np.sqrt( (self.d_0[i]/2*1e-3)  ) * (d*1e-3)**(3/2) ) / (1 - nu**2) + C
            
        
            popt, pcov = curve_fit(Force_Hertz, self.d_0[i]- self.d[i][0:lim_max[i]] , self.F_N[i][0:lim_max[i]] )
            self.Young_modulus.append(popt[0])
            base.append(popt[1])
            
        try:
            os.mkdir(self.outfold + '\\plot_of_fitting')
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
            
        if plot == True:
            n = len(self.d)
            colors = pl.cm.copper_r(np.linspace(0,1,n))
            for i in range(len(self.d)):
                plt.figure()
                ax = plt.axes()
                ax.plot(self.d_0[i] - self.d[i][0:lim_max[i]], self.F_N[i][0:lim_max[i]],color='red', linestyle = '-',linewidth=15,alpha=0.5,label='fitted region')
                ax.plot(self.d_0[i] - self.d[i], self.F_N[i],color=colors[i],linestyle = '',marker = 'o',label='phi = '+ str(self.phi[i]))
                ax.plot(self.d_0[i] - self.d[i], Force_Hertz(self.d_0[i] - self.d[i],self.Young_modulus[i],base[i]),color='red' )
                
                ax.set_xlabel(r'$d_0$ - $d$' + str(' ') + ' [mm]',fontsize=18)
                ax.set_ylabel(r'$F_N$' + str(' ') + ' [N]',fontsize=18)
                ax.tick_params(bottom=True, top=True, left=True, right=False)
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
                ax.tick_params(axis="x", direction="in",labelsize=18)
                ax.tick_params(axis="y", direction="in",labelsize=18)
                ax.tick_params(axis='y', which='minor', direction="in")
                ax.tick_params(axis='x', which='minor', direction="in")
                ax.legend()
                plt.savefig(self.outfold + '\\plot_of_fitting'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent.png',dpi=200,bbox_inches='tight')
                plt.savefig(self.outfold + '\\plot_of_fitting'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent.pdf',dpi=200,bbox_inches='tight')
            
            
            
        return
    
    
    def plot_F_N_vs_d(self,semilogy=False,d_0_subtraction=True,separate=False):
        
        try:
            os.mkdir(self.outfold + '\\plot_F_N_vs_d')
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        n = len(self.d)
        colors = pl.cm.copper_r(np.linspace(0,1,n))
        
        x = []
        y = []
        for i in range(len(self.d)):
            y.append(self.F_N[i])
        
        plt.figure()
        ax = plt.axes()

        if semilogy == True:
            ax.set_yscale("log")
            ax.invert_xaxis()
            
        
        if d_0_subtraction == True:
            for i in range(len(self.d)):
                x.append(self.d_0[i]-self.d[i])
            xlabel = r'$d_0$ - $d$' + str(' ') + ' [mm]'
        else:
            for i in range(len(self.d)):
                x.append(self.d[i])
            xlabel = r'$d$' + str(' ') + ' [mm]'
        
            
        for i in range(len(self.d)):
            ax.plot(x[i],y[i],color=colors[i],label='phi = '+ str(self.phi[i]))
        ax.set_xlabel(xlabel,fontsize=18)
        ax.set_ylabel(r'$F_N$' + str(' ') + ' [N]',fontsize=18)
        ax.tick_params(bottom=True, top=True, left=True, right=False)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax.tick_params(axis="x", direction="in",labelsize=18)
        ax.tick_params(axis="y", direction="in",labelsize=18)
        ax.tick_params(axis='y', which='minor', direction="in")
        ax.tick_params(axis='x', which='minor', direction="in")
        #plt.xlim([np.min(x),np.max(x)])
        ax.legend()
        plt.savefig(self.outfold + '\\plot_F_N_vs_d'+ '\\all_samples_.png',dpi=200,bbox_inches='tight')
        plt.savefig(self.outfold + '\\plot_F_N_vs_d'+ '\\all_samples_.pdf',dpi=200,bbox_inches='tight')
        
        if separate == True:
            for i in range(len(self.d)):
                plt.figure()
                ax = plt.axes()
                ax.plot(x[i],y[i],color=colors[i],label='phi = '+ str(self.phi[i]))
                ax.set_xlabel(xlabel,fontsize=18)
                ax.set_ylabel(r'$F_N$' + str(' ') + ' [N]',fontsize=18)
                ax.tick_params(bottom=True, top=True, left=True, right=False)
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
                ax.tick_params(axis="x", direction="in",labelsize=18)
                ax.tick_params(axis="y", direction="in",labelsize=18)
                ax.tick_params(axis='y', which='minor', direction="in")
                ax.tick_params(axis='x', which='minor', direction="in")
                ax.legend()
                plt.savefig(self.outfold + '\\plot_F_N_vs_d'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent.png',dpi=200,bbox_inches='tight')
                plt.savefig(self.outfold + '\\plot_F_N_vs_d'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent.pdf',dpi=200,bbox_inches='tight')
            

        
        
        
        
        return