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
from scipy.interpolate import BSpline, make_interp_spline
from scipy.signal import savgol_filter



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
        self.strain_rate = []
        
        self.engineering_stress = []
        self.Hertz_engineering_stress=[]
        self.DeltaFsuDeltaS = []
        
        
        self.d_0 = 0
        
        self.input_folder = []
        self.strain= []
        self.poisson = []
        self.DeltaF= []
        self.strain_yealding = []
        self.F_yealding= []
        self.DeltaFDeltaS = []
        self.Theta= []
        self.time = []
        self.dissipated_energy = []
       
        
        
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
    
    def save_results(self,poisson_value = False):
        
        try:
            os.mkdir(self.outfold)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
            
        if (poisson_value == True):
            dict = {'phi': self.phi, 'contact points': self.d_0,'Youngs modulus':self.Young_modulus,'errore Young modulus' :self.Young_modulus_err,'input folder':self.input_folder}
        
            df = pd.DataFrame(dict) 
            df.to_csv(self.outfold + '\\general_informations.csv') 
            
        if (poisson_value == False):
            dict = {'phi': self.phi, 'contact points': self.d_0,'Youngs modulus':self.Young_modulus,'errore Young modulus' :self.Young_modulus_err,'input folder':self.input_folder}
         
            df = pd.DataFrame(dict) 
            df.to_csv(self.outfold + '\\general_informations.csv') 
        
        try:
            os.mkdir(self.outfold + '\\cvs_results_F_N_vs_d0_d')
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        for i in range(len(self.d)):
            dict = {'d0 - d': self.d_0[i] - self.d[i], 'Normal Force': self.F_N[i]}
            df = pd.DataFrame(dict) 
    
            # saving the dataframe 
            df.to_csv(self.outfold + '\\cvs_results_F_N_vs_d0_d' + '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent'+str(i)+'.csv')
            
        
        
        
        if (poisson_value == True):
            print('vero')
            try:
                os.mkdir(self.outfold + '\\cvs_results_all_data')
            except FileExistsError:
                print('directory already existing, graphs will be uploaded')
        
        
            for i in range(len(self.d)):
                dict = {'d0 - d': self.d_0[i] - self.d[i], 'Normal Force': self.F_N[i],'phi': self.phi[i], 'contact points': self.d_0[i],'Youngs modulus':self.Young_modulus[i],'errore Young modulus' :self.Young_modulus_err[i], 'Hertz engineering_stress': self.Hertz_engineering_stress[i], 'strain': self.strain[i],'poisson':self.poisson[i]}
                df = pd.DataFrame(dict) 
        
                # saving the dataframe 
                df.to_csv(self.outfold + '\\cvs_results_all_data' + '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent'+str(i)+'.csv')

        if (poisson_value == False):
            print('falso')
            try:
                os.mkdir(self.outfold + '\\cvs_results_all_data')
            except FileExistsError:
                print('directory already existing, graphs will be uploaded')
        
        
            for i in range(len(self.d)):
                dict = {'d0 - d': self.d_0[i] - self.d[i], 'Normal Force': self.F_N[i],'phi': self.phi[i], 'contact points': self.d_0[i],'Youngs modulus':self.Young_modulus[i],'errore Young modulus' :self.Young_modulus_err[i], 'engineering_stress': self.engineering_stress[i], 'strain': self.strain[i]}
                df = pd.DataFrame(dict) 
        
                # saving the dataframe 
                df.to_csv(self.outfold + '\\cvs_results_all_data' + '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent'+str(i)+'.csv')


        
        return
    
   

    def load_data_from_rhometer(self,path,phi,skip=[], scream_y = False,scream_x = False):
        a = []
        
        self.input_folder.append(path)
        
        #n_raw = pd.read_csv(path, index_col=None,skiprows=5,nrows=1,usecols=[2],sep='\t', decimal=",",encoding='UTF-16 LE')
     
        if len(skip) ==0:
            a.append(pd.read_csv(path, index_col=None,skiprows=10,names= ['Normal Force','time','Distance','strain rate'],usecols=[2,4,5,7],sep='\t', decimal=",",encoding='UTF-16 LE'))
        
        else:
            
            a.append(pd.read_csv(path, index_col=None,skiprows=[0,1,2,3,4,5,6,7,8,9,10]+skip,names= ['Normal Force','time','Distance','strain rate'],usecols=[2,4,5,7],sep='\t', decimal=",",encoding='UTF-16 LE'))

        
        self.F_N.append( np.asarray(a[0]['Normal Force']))
        self.time.append(np.asarray(a[0]['time']))
        self.d.append(np.asarray(a[0]['Distance']))
        self.d_0 = []
        
        self.phi.append(phi)
        
        counter = len(self.F_N)-1
        
        if (scream_y == True):
            self.d[counter], self.F_N[counter],self.time[counter] = sf.excess_xydata_average(self.d[counter], self.F_N[counter],self.time[counter])
        if (scream_x == True):
            self.F_N[counter],self.d[counter],self.time[counter]  = sf.excess_xydata_average(self.F_N[counter], self.d[counter],self.time[counter])
        
        
        
        return
    
    def set_d_0(self,d_0):
        if len(d_0) != len(self.d):
            print('warning: number of d_0 different from the number of experiment')
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
            self.F_N[i] = np.delete(self.F_N[i],lst[i],0)
            self.time[i]= np.delete(self.time[i],lst[i],0)
            
    
        return 
    
    def cut_begin_manual(self,npoints,plot=False):
        
        
        lst = []
        
        for i in range(len(self.d)):
            lst.append(list(range(npoints[i])))
        
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
            self.F_N[i] = np.delete(self.F_N[i],lst[i],0)
            self.time[i]= np.delete(self.time[i],lst[i],0)
        
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
            self.F_N[i] =  np.delete(self.F_N[i],lst[i],0)
            self.time[i]= np.delete(self.time[i],lst[i],0)
            
    
        return 
    
    
    
    def Fit_Hertz_Model(self,nu,lim_max,plot=True):
        
        self.Young_modulus_err = []
        self.Young_modulus = []
        self.base = []
        for i in range(len(self.d)):
            Force_Hertz = lambda d , E , C:  4/3 * ( E * np.sqrt( (self.d_0[i]/2*1e-3)  ) * (d*1e-3)**(3/2) ) / (1 - nu**2) + C
            
        
            popt, pcov = curve_fit(Force_Hertz, self.d_0[i]- self.d[i][0:lim_max[i]] , self.F_N[i][0:lim_max[i]] )
            self.Young_modulus.append(popt[0])
            self.base.append(popt[1])
            self.Young_modulus_err.append( np.sqrt(np.diag(pcov)[1]))
        
        for j in range(len(self.d)):
            self.engineering_stress.append((self.F_N[j])/(self.Young_modulus[j]*((self.d_0[j])**2)))
            self.strain.append((self.d_0[j]-self.d[j])/(self.d_0[j]))
           
          
            
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
                ax.plot(self.d_0[i] - self.d[i], Force_Hertz(self.d_0[i] - self.d[i],self.Young_modulus[i],self.base[i]),color='red' )
                
                '''
                sf.smooth_data(self.d_0[i] - self.d[i], self.F_N[i],9 , 3, ax)
                '''
                
                ax.set_xlabel(r'$d_0$ - $d$' + str(' ') + ' [mm]',fontsize=18)
                ax.set_ylabel(r'$F_N$' + str(' ') + ' [N]',fontsize=18)
                ax.tick_params(bottom=True, top=True, left=True, right=False)
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
                ax.tick_params(axis="x", direction="in",labelsize=18)
                ax.tick_params(axis="y", direction="in",labelsize=18)
                ax.tick_params(axis='y', which='minor', direction="in")
                ax.tick_params(axis='x', which='minor', direction="in")
                ax.legend()
                plt.savefig(self.outfold + '\\plot_of_fitting'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent'+ str(i)+'.png',dpi=200,bbox_inches='tight')
                plt.savefig(self.outfold + '\\plot_of_fitting'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent'+ str(i)+'.pdf',dpi=200,bbox_inches='tight')
            
          
            
        return
    
    def Fit_Hertz_Model_2_0(self,nu,lim_min,lim_max,plot=True):
        
        self.Young_modulus_err = []
        self.Young_modulus = []
        self.base = []
        for i in range(len(self.d)):
            Force_Hertz = lambda d , E , C:  4/3 * ( E * np.sqrt( (self.d_0[i]/2*1e-3)  ) * (d*1e-3)**(3/2) ) / (1 - nu**2) + C
            
            print(i)
            popt, pcov = curve_fit(Force_Hertz, self.d_0[i]- self.d[i][lim_min[i]:lim_max[i]] , self.F_N[i][lim_min[i]:lim_max[i]] )
            self.Young_modulus.append(popt[0])
            self.base.append(popt[1])
            self.Young_modulus_err.append( np.sqrt(np.diag(pcov)[1]))
        
        for j in range(len(self.d)):
            self.engineering_stress.append((self.F_N[j])/(self.Young_modulus[j]*((self.d_0[j])**2)))
            self.strain.append((self.d_0[j]-self.d[j])/(self.d_0[j]))
           
          
            
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
                ax.plot(self.d_0[i] - self.d[i][lim_min[i]:lim_max[i]], self.F_N[i][lim_min[i]:lim_max[i]],color='red', linestyle = '-',linewidth=15,alpha=0.5,label='fitted region')
                ax.plot(self.d_0[i] - self.d[i], self.F_N[i],color=colors[i],linestyle = '',marker = 'o',label='phi = '+ str(self.phi[i]))
                ax.plot(self.d_0[i] - self.d[i][lim_min[i]:lim_max[i]], Force_Hertz(self.d_0[i] - self.d[i][lim_min[i]:lim_max[i]],self.Young_modulus[i],self.base[i]),color='red' )
                
                '''
                sf.smooth_data(self.d_0[i] - self.d[i], self.F_N[i],9 , 3, ax)
                '''
                
                ax.set_xlabel(r'$d_0$ - $d$' + str(' ') + ' [mm]',fontsize=18)
                ax.set_ylabel(r'$F_N$' + str(' ') + ' [N]',fontsize=18)
                ax.tick_params(bottom=True, top=True, left=True, right=False)
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
                ax.tick_params(axis="x", direction="in",labelsize=18)
                ax.tick_params(axis="y", direction="in",labelsize=18)
                ax.tick_params(axis='y', which='minor', direction="in")
                ax.tick_params(axis='x', which='minor', direction="in")
                ax.legend()
                plt.savefig(self.outfold + '\\plot_of_fitting'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent'+ str(i)+'.png',dpi=200,bbox_inches='tight')
                plt.savefig(self.outfold + '\\plot_of_fitting'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent'+ str(i)+'.pdf',dpi=200,bbox_inches='tight')
            
          
            
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
            ax.plot(x[i],y[i],color=colors[i],label=r'$\tilde\epsilon$ = '+ str(self.strain_rate[i]))
        ax.set_xlabel(xlabel,fontsize=18)
        ax.set_ylabel(r'$F_N$' + str(' ') + ' [N]',fontsize=18)
        ax.tick_params(bottom=True, top=True, left=True, right=False)
        ax.set_title(r'Normal Force vs Distance,$\varphi$ = ' + str(int(self.phi[1]*100)) + '%',fontsize=15)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax.tick_params(axis="x", direction="in",labelsize=18)
        ax.tick_params(axis="y", direction="in",labelsize=18)
        ax.tick_params(axis='y', which='minor', direction="in")
        ax.tick_params(axis='x', which='minor', direction="in")
        #plt.xlim([np.min(x),np.max(x)])
        ax.legend(loc=1, prop={'size': 6})
        plt.savefig(self.outfold + '\\plot_F_N_vs_d'+ '\\all_samples_.png',dpi=200,bbox_inches='tight')
        plt.savefig(self.outfold + '\\plot_F_N_vs_d'+ '\\all_samples_.pdf',dpi=200,bbox_inches='tight')
        
        if separate == True:
            for i in range(len(self.d)):
                plt.figure()
                ax = plt.axes()
                ax.plot(x[i],y[i],color=colors[i],label=r'$\tilde\epsilon$ = '+ str(self.phi[i]))
                ax.set_xlabel(xlabel,fontsize=18)
                ax.set_ylabel(r'$F_N$' + str(' ') + ' [N]',fontsize=18)
                ax.set_title(r'Normal Force vs Distance,$\varphi$ = ' + str(int(self.phi[1]*100)) + '%',fontsize=15)
                ax.tick_params(bottom=True, top=True, left=True, right=False)
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
                ax.tick_params(axis="x", direction="in",labelsize=18)
                ax.tick_params(axis="y", direction="in",labelsize=18)
                ax.tick_params(axis='y', which='minor', direction="in")
                ax.tick_params(axis='x', which='minor', direction="in")
                ax.legend(loc=1, prop={'size': 8})
                plt.savefig(self.outfold + '\\plot_F_N_vs_d'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent'+ str(i)+'.png',dpi=200,bbox_inches='tight')
                plt.savefig(self.outfold + '\\plot_F_N_vs_d'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent'+ str(i)+'.pdf',dpi=200,bbox_inches='tight')
            

        
        
        
        
        return
    
    ################################################# EMA ##################################################################
    
    #da sistemare
        self.error_phi = []
        self.phi1 = []
    def carica_err_phi(self,x,y):
        
        self.error_phi.append(y)
        self.phi1.append(x)
        return
    
    
    def save_error_phi(self):
     
        
        try:
            os.mkdir(self.outfold)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
            
            
        dict = {'phi': self.phi1, 'phi error': self.error_phi,'Youngs modulus':self.Young_modulus,'errore Young modulus' :self.Young_modulus_err, 'input folder':self.input_folder}
        
        df = pd.DataFrame(dict) 
    
        # saving the dataframe 
        df.to_csv(self.outfold + '\\errori') 
        
        return
    
    def plot_F_N_su_Fmax_vs_d(self,semilogy=False,d_0_subtraction=True,separate=False):
        
        try:
            os.mkdir(self.outfold + '\\plot_F_N_su_Fmax_vs_d')
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        n = len(self.d)
        colors = pl.cm.copper_r(np.linspace(0,1,n))
        
        x = []
        y = []
        for i in range(len(self.d)):
            y.append(self.F_N[i]/max(self.F_N[i]))
        
        
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
            ax.plot(x[i],y[i],color=colors[i],label=r'$\tilde\epsilon$ = '+ str(self.strain_rate[i]))
        ax.set_xlabel(xlabel,fontsize=18)
        ax.set_ylabel(r'$F_N$/$F_max$' + str(' ') + ' [-]',fontsize=18)
        ax.tick_params(bottom=True, top=True, left=True, right=False)
        
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax.tick_params(axis="x", direction="in",labelsize=18)
        ax.tick_params(axis="y", direction="in",labelsize=18)
        ax.tick_params(axis='y', which='minor', direction="in")
        ax.tick_params(axis='x', which='minor', direction="in")
        #plt.xlim([np.min(x),np.max(x)])
        ax.legend(loc=2, prop={'size': 6})
        plt.savefig(self.outfold + '\\plot_F_N_su_Fmax_vs_d'+ '\\all_samples_.png',dpi=200,bbox_inches='tight')
        plt.savefig(self.outfold + '\\plot_F_N_su_Fmax_vs_d'+ '\\all_samples_.pdf',dpi=200,bbox_inches='tight')
        
        if separate == True:
            for i in range(len(self.d)):
                plt.figure()
                ax = plt.axes()
                ax.plot(x[i],y[i],color=colors[i],label=r'$\tilde\epsilon$ = '+ str(self.strain_rate[i]))
                ax.set_xlabel(xlabel,fontsize=18)
                ax.set_ylabel(r'$F_N$/$F_M$' + str(' ') + ' [-]',fontsize=18)
                ax.set_title(r'Normal Force vs Distance,$\varphi$ = ' + str(int(self.phi[i]*100)) + '%',fontsize=15)
                ax.tick_params(bottom=True, top=True, left=True, right=False)
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
                ax.tick_params(axis="x", direction="in",labelsize=18)
                ax.tick_params(axis="y", direction="in",labelsize=18)
                ax.tick_params(axis='y', which='minor', direction="in")
                ax.tick_params(axis='x', which='minor', direction="in")
                ax.legend(loc=2, prop={'size': 8})
                plt.savefig(self.outfold + '\\plot_F_N_su_Fmax_vs_d'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent'+ str(i)+'.png',dpi=200,bbox_inches='tight')
                plt.savefig(self.outfold + '\\plot_F_N_su_Fmax_vs_d'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent'+ str(i)+'.pdf',dpi=200,bbox_inches='tight')
        
        

    
    def plot_F_N_su_Hertz_vs_strain(self,semilogy=False,d_0_subtraction=True,separate=False):
        
        try:
            os.mkdir(self.outfold + '\\plot_F_N_su_Hertz_vs_strain')
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        try:
            os.mkdir(self.outfold + '\\plot_F_N_su_Hertz_vs_d0^2')
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        n = len(self.d)
        colors = pl.cm.winter_r(np.linspace(0,1,n))
        
        x = []
        y = []
        for i in range(len(self.d_0)):
            y.append((self.F_N[i])/(self.Young_modulus[i]*((self.d_0[i]*1e-03)**2)))
        
        
        plt.figure()
        ax = plt.axes()

        if semilogy == True:
            ax.set_yscale("log")
            ax.invert_xaxis()
            
        
        if d_0_subtraction == True:
            for i in range(len(self.d)):
                x.append((self.d_0[i]-self.d[i])/self.d_0[i])
            xlabel = r'$d_0$ - $d$' + str(' ') + ' [mm]'
        else:
            for i in range(len(self.d)):
                x.append(self.d[i])
            xlabel = r'$d$' + str(' ') + ' [mm]'
        
            
        for i in range(len(self.d)):
            ax.plot(x[i],y[i],color=colors[i],label='phi = '+ str(self.phi[i]))
        ax.set_xlabel(r'$\epsilon$',fontsize=18)
        ax.set_ylabel(r'$F_N$/$E*$' + str(' ') + ' [-]',fontsize=18)
        ax.tick_params(bottom=True, top=True, left=True, right=False)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax.tick_params(axis="x", direction="in",labelsize=18)
        ax.tick_params(axis="y", direction="in",labelsize=18)
        ax.tick_params(axis='y', which='minor', direction="in")
        ax.tick_params(axis='x', which='minor', direction="in")
        #plt.xlim([np.min(x),np.max(x)])
        ax.legend()
        plt.savefig(self.outfold + '\\plot_F_N_su_Hertz_vs_strain'+ '\\all_samples_.png',dpi=200,bbox_inches='tight')
        plt.savefig(self.outfold + '\\plot_F_N_su_Hertz_vs_strain'+ '\\all_samples_.pdf',dpi=200,bbox_inches='tight')
        
        if separate == True:
            for i in range(len(self.d)):
                plt.figure()
                ax = plt.axes()
                ax.plot(x[i],y[i],color=colors[i],label='phi = '+ str(self.phi[i]))
                ax.set_xlabel(xlabel,fontsize=18)
                ax.set_ylabel(r'$F/(E^*\dot (d_0)^2)$' + str(' ') + ' [-]',fontsize=18)
                ax.tick_params(bottom=True, top=True, left=True, right=False)
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
                ax.tick_params(axis="x", direction="in",labelsize=18)
                ax.tick_params(axis="y", direction="in",labelsize=18)
                ax.tick_params(axis='y', which='minor', direction="in")
                ax.tick_params(axis='x', which='minor', direction="in")
                ax.legend()
                plt.savefig(self.outfold + '\\plot_F_N_su_Hertz_vs_d0^2'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent'+ str(i)+'.png',dpi=200,bbox_inches='tight')
                plt.savefig(self.outfold + '\\plot_F_N_su_Hertz_vs_d0^2'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent'+ str(i)+'.pdf',dpi=200,bbox_inches='tight')

        

    def Fit_Poisson_and_Young(self,lim_max,nu=None,Young_mod=None,plot=True):
            
        self.costant_Hertz=[]
        self.costant_Hertz_err = []
        self.strain=[]
        
        for i in range(len(self.d)):
            Force_Hertz = lambda d , m :  4/3 * (m * np.sqrt( (self.d_0[i]/2*1e-3)  ) * (d*1e-3)**(3/2) )  
                
            # dove m = E/(1-nu^2)
            popt, pcov = curve_fit(Force_Hertz, self.d_0[i]- self.d[i][0:lim_max[i]] , self.F_N[i][0:lim_max[i]] ,bounds=([Young_mod],[10*Young_mod/3]))
            self.costant_Hertz.append(popt[0])
            
            self.costant_Hertz_err.append(np.sqrt(np.diag(pcov)[0]))
        
        for j in range(len(self.d)):
            #self.engineering_stress.append((self.F_N[j])/(self.Young_modulus[j]*((self.d_0[j])**2)))
            self.strain.append((self.d_0[j]-self.d[j])/(self.d_0[j]))
            self.Hertz_engineering_stress.append((self.F_N[j])/(self.costant_Hertz[j]*((self.d_0[j]*1e-03)**2)))
        
        if(Young_mod==None):

            if (nu==None):
                print ('put a value at nu, operation not found')
                
            else:
                self.Young_modulus = []
                self.poisson= []
                for i in range(len(self.d)):
                    self.poisson.append(nu)
                    self.Young_modulus.append(self.costant_Hertz[i]*(1-nu[i]**2))

        if(nu==None):
            if (Young_mod==None):
                print ('put a value at Young_mod, operation not found')
                
            else:
                self.Young_modulus = []
                self.poisson= []
                for i in range(len(self.d)):
                    self.Young_modulus.append(Young_mod)
                    self.poisson.append(np.sqrt(np.abs(1-(self.Young_modulus[i]/self.costant_Hertz[i]))))
            
        try:
            os.mkdir(self.outfold + '\\plot_of_fitting_Poisson_and_Young')
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
                
        if plot == True:
            n = len(self.d)
            colors = pl.cm.copper(np.linspace(0,1,n))
            for i in range(len(self.d)):
                plt.figure()
                ax = plt.axes()
                ax.plot((self.d_0[i] - self.d[i][0:lim_max[i]]), self.F_N[i][0:lim_max[i]],color='red', linestyle = '-',linewidth=15,alpha=0.5,label='fitted region')
                ax.plot((self.d_0[i] - self.d[i]), self.F_N[i],color=colors[i],linestyle = '',marker = 'o',label='phi = '+ str(self.phi[i]))
                ax.plot((self.d_0[i] - self.d[i]), Force_Hertz(self.d_0[i] - self.d[i],self.costant_Hertz[i]),color='red' )
                
                '''
                sf.smooth_data(self.d_0[i] - self.d[i], self.F_N[i],9 , 3, ax)
                '''
                
                ax.set_xlabel(r'$d_0 - d$' + str(' ') + ' [-]',fontsize=18)
                ax.set_ylabel(r'$F_N$' + str(' ') + ' [N]',fontsize=18)
                ax.tick_params(bottom=True, top=True, left=True, right=False)
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
                ax.tick_params(axis="x", direction="in",labelsize=18)
                ax.tick_params(axis="y", direction="in",labelsize=18)
                ax.tick_params(axis='y', which='minor', direction="in")
                ax.tick_params(axis='x', which='minor', direction="in")
                ax.legend()
                plt.savefig(self.outfold + '\\plot_of_fitting_Poisson_and_Young'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent'+ str(i)+'.png',dpi=200,bbox_inches='tight')
                plt.savefig(self.outfold + '\\plot_of_fitting_Poisson_and_Young'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent'+ str(i)+'.pdf',dpi=200,bbox_inches='tight')
        
     
          
            
        return  
    
    def plot_normalized_stress_vs_strain_with_ideal_contact(self,semilogy=False,xlim=1.0,ylim=200000,separate=False):
        
        n = len(self.d)
        colors = pl.cm.tab10_r(np.linspace(0,1,n))
        
        x = []
        y = []
        fake_force = []
        
        E_average = np.mean(np.asarray(self.costant_Hertz))
        strain_fake = np.linspace(0,1,1000)
        
        for j in range(len(self.d_0)):
            
            y.append((self.F_N[j])/(self.d_0[j]*1e-03)**2)
            x.append((self.d_0[j]-self.d[j])/self.d_0[j])
            
            fake_force.append( 4/3*self.costant_Hertz[j]*np.sqrt(self.d_0[j]*1e-03/2) * (self.d_0[j]*1e-03 * strain_fake)**(3/2) / (self.d_0[j]*1e-03)**2)
        
        
        
        
        

        for i in range(len(self.d)):
            plt.figure()
            ax = plt.axes()
            ax.plot(x[i],y[i],color=colors[i],label=r'$\tilde\epsilon$ = '+ str(np.round(self.strain_rate[i],5)))
            ax.plot(strain_fake,fake_force[i],color='red')
            ax.set_xlabel(r'$\epsilon$',fontsize=18)
            ax.set_ylabel(r'$\sigma_N$' + str(' ') + ' [Pa]',fontsize=18)
            ax.set_title(r'Engineering stress vs strain, $\varphi$ =' + str(int(self.phi[1]*100))+'%',fontsize=18) 
            ax.tick_params(bottom=True, top=True, left=True, right=False)
            ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
            ax.tick_params(axis="x", direction="in",labelsize=18)
            ax.tick_params(axis="y", direction="in",labelsize=18)
            ax.tick_params(axis='y', which='minor', direction="in")
            ax.tick_params(axis='x', which='minor', direction="in")
            #plt.xlim([np.min(x),np.max(x)])
            ax.legend(loc=1, prop={'size': 6})
            ax.set_xlim([0,xlim])
            ax.set_ylim([0,ylim])
        #plt.savefig(self.outfold + '\\plot_F_N_su_Costant_Hertz_vs_strain'+ '\\all_samples_.png',dpi=200,bbox_inches='tight')
        #plt.savefig(self.outfold + '\\plot_F_N_su_Costant_Hertz_vs_strain'+ '\\all_samples_.pdf',dpi=200,bbox_inches='tight')
        
        
        return
    
    def Dissipated_Energy(self,strain_plasticity,strain_yielding,xlim=1.0,ylim=200000):
        
        interpolated_data = []
        self.dissipated_energy = []
        n = len(self.d)
        
        self.DeltaFsuDeltaS = []
        
        for i in range(n):
            self.DeltaFsuDeltaS.append(0)
            self.DeltaF.append(0)
            self.strain_yealding.append(0)
            self.F_yealding.append(0)
            self.Theta.append(0)
        

        fake_force = []
        
        idx_p = []
        idx_y = []
        
        
        
        for j in range(n):
            
            new_x = np.linspace(np.min(self.d_0[j]-self.d[j]),np.max(self.d_0[j]-self.d[j]),10000)
            
            y_intepolated = interpolate.interp1d(self.d_0[j]-self.d[j], (self.F_N[j]))
            interpolated_data.append(y_intepolated)
            
            fake_force = 4/3*self.costant_Hertz[j]*np.sqrt(self.d_0[j]*1e-03/2) *  (new_x*1e-03)**(3/2) 
            fake_interpolated = interpolate.interp1d(new_x,fake_force)
            

            idx_p = sf.find_nearest(new_x, strain_plasticity[j]*self.d_0[j])
            
            
            idx_y = sf.find_nearest(new_x, strain_yielding[j]*self.d_0[j])
            
            
            area_h =  integrate.quad(fake_interpolated,new_x[idx_p],new_x[idx_y])
            area_data =  integrate.quad(y_intepolated,new_x[idx_p],new_x[idx_y])
            
            
                
            area = (area_h[0] - area_data[0])*1e-03
            
                
            #(fake_interpolated(new_x[idx_p:idx_y]) - y_intepolated(new_x[idx_p:idx_y])) * new_x[idx_p:idx_y]
            self.dissipated_energy.append(area)
            
            plt.figure()
            ax = plt.axes()
            ax.plot(new_x, y_intepolated(new_x))
            ax.plot(new_x,  fake_interpolated(new_x),color='red')
            ax.fill_between(new_x[idx_p:idx_y], fake_interpolated(new_x[idx_p:idx_y]), y_intepolated(new_x[idx_p:idx_y]),alpha=0.3)
            
            ax.vlines(x=strain_plasticity[j]*self.d_0[j], ymin=0, ymax=ylim)
            ax.vlines(x=strain_yielding[j]*self.d_0[j], ymin=0, ymax=ylim)
            ax.set_xlim([0,xlim])
            ax.set_ylim([0,ylim])
            ax.set_xlabel(r'$d_0-d$'+ str(' ') + ' [mm]',fontsize=18)
            ax.set_ylabel(r'$F_N$' + str(' ') + ' [N]',fontsize=18)
             
            ax.tick_params(bottom=True, top=True, left=True, right=False)
            ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
            ax.tick_params(axis="x", direction="in",labelsize=18)
            ax.tick_params(axis="y", direction="in",labelsize=18)
            ax.tick_params(axis='y', which='minor', direction="in")
            ax.tick_params(axis='x', which='minor', direction="in")
            #plt.xlim([np.min(x),np.max(x)])
            ax.legend(loc=1, prop={'size': 6})
        
        '''
        for j in range(n):
           
            # area = ( fake_stress[j](new_x[idx_p[j]:idx_y[j]]) - interpolated_data[j](new_x[idx_p[j]:idx_y[j]]) )* new_x[idx_p[j]:idx_y[j]]

            area =  fake_stress[j][idx_p[j]:idx_y[j]] - interpolated_data[j](new_x[idx_p[j]:idx_y[j]]) 
            dissipated_energy.append(area)
            
        for j in range(n):            
            plt.figure()
            ax = plt.axes()
            ax.plot(new_x, interpolated_data[j](new_x))
            ax.plot(new_x,  fake_stress[j],color='red')
            ax.fill_between(new_x[idx_p[j]:idx_y[j]], fake_stress[j][idx_p[j]:idx_y[j]], interpolated_data[j](new_x[idx_p[j]:idx_y[j]]),alpha=0.3)
            
            ax.vlines(x=strain_plasticity[j], ymin=0, ymax=150000)
            ax.vlines(x=strain_yielding[j], ymin=0, ymax=150000)
            ax.set_xlim([0,xlim])
            ax.set_ylim([0,ylim])

        '''

        return 

    def plot_F_N_su_Costant_Hertz_vs_strain(self,semilogy=False,d_0_subtraction=True,separate=False):
        
        try:
            os.mkdir(self.outfold + '\\plot_F_N_su_Costant_Hertz_vs_strain')
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        try:
            os.mkdir(self.outfold + '\\plot_F_N_su_Costant_Hertz_vs_d0^2')
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        n = len(self.d)
        colors = pl.cm.tab10_r(np.linspace(0,1,n))
        
        x = []
        y = []
        for j in range(len(self.d_0)):
            y.append((self.F_N[j])/(self.costant_Hertz[j]*((self.d_0[j]*1e-03)**2)))
        
        
        plt.figure()
        ax = plt.axes()

        if semilogy == True:
            ax.set_yscale("log")
            ax.invert_xaxis()
            
        
        if d_0_subtraction == True:
            for i in range(len(self.d)):
                x.append((self.d_0[i]-self.d[i])/self.d_0[i])
            xlabel = r'$\epsilon$' + str(' ') + ' [-]'
        else:
            for i in range(len(self.d)):
                x.append(self.d[i])
            xlabel = r'$d$' + str(' ') + ' [mm]'
        
            
        for i in range(len(self.d)):
            ax.plot(x[i],y[i],color=colors[i],label=r'$\tilde\epsilon$ = '+ str(np.round(self.strain_rate[i],5)))
        ax.set_xlabel(r'$\epsilon$',fontsize=18)
        ax.set_ylabel(r'$F/(E^*\dot (d_0)^2)$' + str(' ') + ' [-]',fontsize=18)
        ax.set_title(r'Engineering stress vs strain, $\varphi$ =' + str(int(self.phi[1]*100))+'%',fontsize=18) 
        ax.tick_params(bottom=True, top=True, left=True, right=False)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax.tick_params(axis="x", direction="in",labelsize=18)
        ax.tick_params(axis="y", direction="in",labelsize=18)
        ax.tick_params(axis='y', which='minor', direction="in")
        ax.tick_params(axis='x', which='minor', direction="in")
        #plt.xlim([np.min(x),np.max(x)])
        ax.legend(loc=1, prop={'size': 6})
        plt.savefig(self.outfold + '\\plot_F_N_su_Costant_Hertz_vs_strain'+ '\\all_samples_.png',dpi=200,bbox_inches='tight')
        plt.savefig(self.outfold + '\\plot_F_N_su_Costant_Hertz_vs_strain'+ '\\all_samples_.pdf',dpi=200,bbox_inches='tight')
        
        if separate == True:
            for i in range(len(self.d)):
                plt.figure()
                ax = plt.axes()
                ax.plot(x[i],y[i],color=colors[i],label=r'$\tilde\epsilon$ = '+ str(self.strain_rate[i]))
                ax.set_xlabel(xlabel,fontsize=18)
                ax.set_ylabel(r'$F/(E^*\dot (d_0)^2)$' + str(' ') + ' [-]',fontsize=18)
                ax.set_title(r'Engineering stress vs strain, $\varphi$ =' + str(int(self.phi[i]*100))+'%',fontsize=18) 
                ax.tick_params(bottom=True, top=True, left=True, right=False)
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
                ax.tick_params(axis="x", direction="in",labelsize=18)
                ax.tick_params(axis="y", direction="in",labelsize=18)
                ax.tick_params(axis='y', which='minor', direction="in")
                ax.tick_params(axis='x', which='minor', direction="in")
                ax.legend(loc=2, prop={'size': 8.5})
                plt.savefig(self.outfold + '\\plot_F_N_su_Costant_Hertz_vs_d0^2'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent'+ str(i)+'.png',dpi=200,bbox_inches='tight')
                plt.savefig(self.outfold + '\\plot_F_N_su_Costant_Hertz_vs_d0^2'+ '\\sample_volume_fraction_'+str(int(self.phi[i]*100)) +'_percent'+ str(i)+'.pdf',dpi=200,bbox_inches='tight')
                
    
    def Reset_theta_yealding(self):
            self.DeltaF= []
            self.strain_yealding = []
            self.F_yealding= []
            self.DeltaFsuDeltaS = []
            self.Theta= [] 
            self.DeltaS=[]
    
    def Delta_F_delta_strain(self,list_element,index_in,index_fin,ind_max_in,ind_max_fin,stocazzo=True,plot=True):
        n=len(self.d)
        colors = pl.cm.inferno_r(np.linspace(0,1,n))
        
        try:
            os.mkdir(self.outfold + '\\plot_theta')
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
            
       
        
        self.Hertz_engineering_stress[list_element]=np.round(self.Hertz_engineering_stress[list_element],15)
        
        engineering_stress_max = np.max(np.asarray(self.Hertz_engineering_stress[list_element][ind_max_in:ind_max_fin]))
        
        engineering_stress_min = np.min(np.asarray(self.Hertz_engineering_stress[list_element][index_in:index_fin]))
        
        
        strain_max_ind, = np.where(self.Hertz_engineering_stress[list_element][ind_max_in:ind_max_fin]== engineering_stress_max)
        strain_min_ind, = np.where(self.Hertz_engineering_stress[list_element][index_in-1:index_fin+1]== engineering_stress_min)
        
        if(plot == True): 
        
            plt.figure()
            ax = plt.axes()
            ax.plot(self.strain[list_element][index_fin], self.Hertz_engineering_stress[list_element][index_fin],color='red', linestyle = '',marker = '<',label='right bound',markersize=12)
            ax.plot(self.strain[list_element][index_in], self.Hertz_engineering_stress[list_element][index_in],color='red', linestyle = '',marker = '>',label='left bound',markersize=12)
            ax.plot(self.strain[list_element], self.Hertz_engineering_stress[list_element],color='blue',linestyle = '',marker = 'o')
            ax.plot([self.strain[list_element][strain_min_ind[0]+index_in-1],self.strain[list_element][strain_max_ind[0]+ind_max_in]],[engineering_stress_min,engineering_stress_max],color=colors[list_element],linestyle = '-',marker = 'o',markersize=14,linewidth=5)
            
            
            
        
            ax.set_title((r'$\theta$ for $\tilde\epsilon$=' +str(np.round(self.strain_rate[list_element],4))),size=16)
            ax.set_xlabel(r'$\epsilon$' + str(' ') + ' [-]',fontsize=18)
            ax.set_ylabel(r'$F/(E^*\dot (d_0)^2)$' + str(' ') + ' [-]',fontsize=18)
            ax.tick_params(bottom=True, top=True, left=True, right=False)
            ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
            ax.tick_params(axis="x", direction="in",labelsize=18)
            ax.tick_params(axis="y", direction="in",labelsize=18)
            ax.tick_params(axis='y', which='minor', direction="in")
            ax.tick_params(axis='x', which='minor', direction="in")
            ax.legend(loc=4, prop={'size': 8})
           
            plt.savefig(self.outfold + '\\plot_theta'+ '\\sample_volume_fraction_'+str(int(self.phi[list_element]*100)) +'_percent'+ str(list_element)+'.png',dpi=200,bbox_inches='tight')
            plt.savefig(self.outfold + '\\plot_theta'+ '\\sample_volume_fraction_'+str(int(self.phi[list_element]*100)) +'_percent'+ str(list_element)+'.pdf',dpi=200,bbox_inches='tight')
        
        self.DeltaF.append(engineering_stress_max-engineering_stress_min)
        self.strain_yealding.append(self.strain[list_element][strain_max_ind[0]+ind_max_in])
        self.F_yealding.append(engineering_stress_max)
        self.DeltaFsuDeltaS.append((engineering_stress_max-engineering_stress_min)/(self.strain[list_element][strain_max_ind[0]]-self.strain[list_element][strain_min_ind[0]]))
        self.Theta.append(np.arctan(np.abs(((engineering_stress_max-engineering_stress_min)/(self.strain[list_element][strain_max_ind[0]]-self.strain[list_element][strain_min_ind[0]])))))
        self.DeltaS.append(self.strain[list_element][strain_max_ind[0]]-self.strain[list_element][strain_min_ind[0]])
        
        return (engineering_stress_max-engineering_stress_min)/(self.strain[list_element][strain_max_ind[0]]-self.strain[list_element][strain_min_ind[0]])

    def set_strain_rate(self,rate):
        
        for i in range(len(self.d_0)):
            self.strain_rate.append(rate[i])
        return
    
    
    def save_results_new(self):
        '''
        try:
            os.mkdir(self.resume_fold )
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        dict = {'phi [-]': self.phi, 'strain rate [1/s]':self.strain_rate ,'d_0 [mm]': self.d_0,'costant Hertz [Pa]':self.costant_Hertz,'E [Pa]':self.Young_modulus, 'poisson ratio [-]':self.poisson,'yealding strain [-]':self.strain_yealding,'yealding force [-]':self.F_yealding,'Delta Force[-]':self.DeltaF,'Delta Force su Delta strain [-]':self.DeltaFsuDeltaS,r'theta [rad]':self.Theta}
        
        df = pd.DataFrame(dict) 
        df.to_csv(self.resume_fold +'\\results_' + str(int(self.phi[0]*100))+'vf' + '_final_informations.csv') 
        '''
        try:
            os.mkdir(self.resume_fold+ '\\all_data'+ str(int(self.phi[0]*100)) )
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        for i in range(len(self.d)):
            
            try:
                dict = {'d_0-d [mm]': self.d_0[i]-self.d[i], 'Normal Force [N]': self.F_N[i],'phi[-]': self.phi[i], 'contact points[mm]': self.d_0[i], 'Hertz engineering_stress [-]': self.Hertz_engineering_stress[i], 'strain [-]': self.strain[i], 'time [s]': self.time[i]}
                df = pd.DataFrame(dict) 
            
                # saving the dataframe 
                df.to_csv(self.resume_fold+ '\\all_data'+ str(int(self.phi[0]*100)) +'\\'+ str(int(self.phi[0]*100))+'vf_cvs_results_all_data_rate_'+ str(self.strain_rate[i]) +'_'+ str(int(i)) +'.csv' )
            except:
                 print(len(self.d_0[i]-self.d[i]))
                 print(len(self.time[i]))
        
        return
    
    def fodel_for_saving_resume(self,name):
        
        self.resume_fold = name
        
        try:
            os.mkdir(self.resume_fold)
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        return
    
    
def interpolation(x,y,number):
        
    smoothed = savgol_filter(y, number, 3)
        
    interp_rate_1    = make_interp_spline(x,smoothed)
    dc = np.linspace(x[0],x[-1],1000)
    return dc,interp_rate_1 