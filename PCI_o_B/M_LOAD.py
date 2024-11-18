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



class MULTI_LOAD():
    
    def __init__(self):
        """Initialize ROIs
        
        Parameters
        ----------
        Filename: complete filename of the CI file
        
        
        """
        
        self.ncycle = []
        self.input_folder = []
        
        self.F_N = []
        self.time  = []
        self.d  = []
        
        self.F_N_unload = []
        self.time_unload  = []
        self.d_unload  = []
        
        self.phi = 0
        
        self.d_0 = []
        
        
        
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
            
            
        dict = {'cycle number ': self.ncycle, 'contact points': self.d_0,'Youngs modulus':self.Young_modulus,'energy stored':self.energy_stored,'input folder':self.input_folder}
        
        df = pd.DataFrame(dict) 
    
        # saving the dataframe 
        df.to_csv(self.outfold + '\\general_informations.csv') 
        
        try:
            os.mkdir(self.outfold + '\\cvs_results_F_N_vs_d_load')
            os.mkdir(self.outfold + '\\cvs_results_F_N_vs_d_unload')
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        for i in range(len(self.d)):
            dict = {'d_load': self.d[i], 'Normal Force load': self.F_N[i]}
            df = pd.DataFrame(dict) 
            
            # saving the dataframe 
            df.to_csv(self.outfold + '\\cvs_results_F_N_vs_d_load' + '\\cycle_number_'+str(i+1) +'.csv')
            
        for i in range(len(self.d)):
            dict = {'d_unload':  self.d_unload[i], 'Normal Force unload': self.F_N_unload[i]}
            df = pd.DataFrame(dict) 
            
            # saving the dataframe 
            df.to_csv(self.outfold + '\\cvs_results_F_N_vs_d_unload' + '\\cycle_number_'+str(i+1) +'.csv')
            
            
        
        
        return


    def load_data_from_rhometer(self,path,nloads,phi,skip=[]):
        
        
        
        for i in range(nloads):
            self.ncycle.append(i+1)
            self.input_folder.append(path)
        
        
        
        
        for j in range(nloads):
            a = []
            if j == 0:
            
     
                if len(skip) == 0:
                    a.append(pd.read_csv(path, index_col=None,skiprows=10,nrows=500,names= ['time','Normal Force','Distance'],usecols=[2,3,5],sep='\t', decimal=",",encoding='UTF-16 LE'))
                
                else:
                    
                    a.append(pd.read_csv(path, index_col=None,skiprows=[0,1,2,3,4,5,6,7,8,9,10]+skip,nrows=500-len(skip)-1,names= ['time','Normal Force','Distance'],usecols=[2,3,5],sep='\t', decimal=",",encoding='UTF-16 LE'))
                    
                self.F_N.append( np.asarray(a[0]['Normal Force']))
                self.time.append(np.asarray(a[0]['time']))
                self.d.append(np.asarray(a[0]['Distance']))
    
            else:
                
                a.append(pd.read_csv(path, index_col=None,skiprows=10+j*(500+5),nrows=500,names= ['time','Normal Force','Distance'],usecols=[2,3,5],sep='\t', decimal=",",encoding='UTF-16 LE'))

            
                self.F_N.append( np.asarray(a[0]['Normal Force']))
                self.time.append(np.asarray(a[0]['time']))
                self.d.append(np.asarray(a[0]['Distance']))
        
        self.d_0 = []
        
        self.phi = phi
        
        return 
    
    def select_unload_curve(self,start,npoints,plot=False):
        
        
        if len(npoints) != len(self.d):
            print('worning: list of points to cut has different lenght from the number of experiment')
            print('no operation has been done')
            return
        
        lst = []
        
        n = len(self.d)
        colors = pl.cm.copper_r(np.linspace(0,1,n))
        
        for i in range(len(self.d)):
            lst.append(list(range(start[i],npoints[i])))
        
            if plot == True:
                plt.figure()
                ax = plt.axes()
                ax.semilogx(self.d[i],self.F_N[i],color = colors[i],marker='.',linestyle='')
                ax.semilogx(self.d[i][lst[i]],self.F_N[i][lst[i]],marker='.',linestyle='',color='red',label=' load n: = '+ str(i+1) + ', unlod curve '+str(npoints[i])+' points')
                ax.set_xlabel(r'$d$' + str(' ') + ' [mm]',fontsize=18)
                ax.set_ylabel(r'$F_N$' + str(' ') + ' [N]',fontsize=18)
                ax.invert_xaxis()
                ax.legend(loc='upper right')
                
            self.d_unload.append(self.d[i][lst[i]])
            self.F_N_unload.append(self.F_N[i][lst[i]])
            
            self.d[i] = np.delete(self.d[i],lst[i],0)
            self.F_N[i] =    np.delete(self.F_N[i],lst[i],0)
        
        
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
    
    
    def cut_begin(self,plot=False):
        
        lst = []
        
        n = len(self.d)
        colors = pl.cm.copper_r(np.linspace(0,1,n))
        
        for i in range(len(self.d)):
            lst.append([i for i,v in enumerate(self.d_0[i] - self.d[i]) if v < 0])
        
       
        
       
        
            if plot == True:
                plt.figure()
                ax = plt.axes()
                ax.semilogx(self.d[i],self.F_N[i],color=colors[i],marker='.',linestyle='')
                ax.semilogx(self.d[i][lst[i]],self.F_N[i][lst[i]],marker='.',linestyle='',color='red',label=' load n: = '+ str(i+1)+ ', delete '+str(len(lst[i]))+' points')
                ax.set_xlabel(r'$d$' + str(' ') + ' [mm]',fontsize=18)
                ax.set_ylabel(r'$F_N$' + str(' ') + ' [N]',fontsize=18)
                ax.invert_xaxis()
                ax.legend(loc='upper right')
                
            self.d[i] = np.delete(self.d[i],lst[i],0)
            self.F_N[i] =    np.delete(self.F_N[i],lst[i],0)
            
            
    
        return 
    
    
    def cut_tails_unload_curve(self,npoints,plot=False):
        
        if len(npoints) != len(self.d):
            print('worning: list of points to cut has different lenght from the number of experiment')
            print('no operation has been done')
            return
        
        lst = []
        
        n = len(self.d)
        colors = pl.cm.copper_r(np.linspace(0,1,n))
        
        for i in range(len(self.d)):
            lst.append(list(range(len(self.d_unload[i])-npoints[i],len(self.d_unload[i]))))
            if len(self.F_N_unload[i]) != 0:
                if plot == True:
                    plt.figure()
                    ax = plt.axes()
                    ax.semilogx(self.d_unload[i],self.F_N_unload[i],color=colors[i],marker='.',linestyle='')
                    ax.semilogx(self.d_unload[i][lst[i]],self.F_N_unload[i][lst[i]],marker='.',linestyle='',color='red',label=' load n: = '+ str(i+1) + ', delete '+str(npoints[i])+' points')
                    ax.set_xlabel(r'$d$' + str(' ') + ' [mm]',fontsize=18)
                    ax.set_ylabel(r'$F_N$' + str(' ') + ' [N]',fontsize=18)
                    ax.invert_xaxis()
                    ax.legend(loc='upper right')
                    
                self.d_unload[i] = np.delete(self.d_unload[i],lst[i],0)
                self.F_N_unload[i] =    np.delete(self.F_N_unload[i],lst[i],0)
            else:
                print('no unload curve for the cycle: '+ str(i+1))
            
            
    
        return 
    
    def cut_tails_load_curve(self,npoints,plot=False):
        
        if len(npoints) != len(self.d):
            print('worning: list of points to cut has different lenght from the number of experiment')
            print('no operation has been done')
            return
        
        lst = []
        
        n = len(self.d)
        colors = pl.cm.copper_r(np.linspace(0,1,n))
        
        for i in range(len(self.d)):
            lst.append(list(range(len(self.d[i])-npoints[i],len(self.d[i]))))
            if len(self.F_N_unload[i]) != 0:
                if plot == True:
                    plt.figure()
                    ax = plt.axes()
                    ax.semilogx(self.d[i],self.F_N[i],color=colors[i],marker='.',linestyle='')
                    ax.semilogx(self.d[i][lst[i]],self.F_N[i][lst[i]],marker='.',linestyle='',color='red',label=' load n: = '+ str(i+1) + ', delete '+str(npoints[i])+' points')
                    ax.set_xlabel(r'$d$' + str(' ') + ' [mm]',fontsize=18)
                    ax.set_ylabel(r'$F_N$' + str(' ') + ' [N]',fontsize=18)
                    ax.invert_xaxis()
                    ax.legend(loc='upper right')
                    
                self.d[i] = np.delete(self.d[i],lst[i],0)
                self.F_N[i] =    np.delete(self.F_N[i],lst[i],0)
            else:
                print('no unload curve for the cycle: '+ str(i+1))
            
            
    
        return 
    
    
    
    def Fit_Hertz_Model(self,nu,lim_max,plot=True):
        
        self.Young_modulus = []
        base = []
        for i in range(len(self.d)):
            print(i)
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
                ax.plot(self.d_0[i] - self.d[i], self.F_N[i],color=colors[i],linestyle = '',marker = 'o',label=' load n: = '+ str(i+1))
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
                
                plt.savefig(self.outfold + '\\plot_of_fitting'+ '\\cycle_number_'+ str(i+1) +'.png',dpi=200,bbox_inches='tight')
                plt.savefig(self.outfold + '\\plot_of_fitting'+ '\\cycle_number_'+ str(i+1) +'.pdf',dpi=200,bbox_inches='tight')
            
            
            
        return
    
    def integral_energy_stored(self,plot=False):
        n = len(self.d)
        colors = pl.cm.copper_r(np.linspace(0,1,n))
        colors_unload = pl.cm.cool(np.linspace(0,1,n))
        
        self.energy_stored = []
        
        try:
            os.mkdir(self.outfold + '\\plot_of_hysteresys')
        except FileExistsError:
            print('directory already existing, graphs will be uploaded')
        
        for i in range(len(self.d)):
            if len(self.F_N_unload[i]) != 0:
                load = interpolate.interp1d(self.d[i],self.F_N[i])
                energy_given = integrate.quad(load , self.d[i][-1], self.d[i][0])
                
                unload = interpolate.interp1d(self.d_unload[i],self.F_N_unload[i])
                energy_released = integrate.quad(unload , self.d_unload[i][0], self.d_unload[i][-1])

                min_d = np.min(self.d[i])
                min_d_uload = np.min(self.d_unload[i])
                
                good_min = max(min_d,min_d_uload)
                
                max_d = np.max(self.d[i])
                max_d_uload = np.max(self.d_unload[i])
                
                
                good_max = min(max_d,max_d_uload)
                
                new_d = np.linspace(good_min, good_max,100)
                
                self.energy_stored.append( energy_given[0] - energy_released[0])
                
                if plot == True:
                    plt.figure()
                    ax = plt.axes()
                    ax.plot(np.flip(new_d-new_d[0]),load(new_d),marker='o',linestyle='',color='green',label='load')
                    ax.plot(np.flip(new_d-new_d[0]),unload(new_d),marker='s',linestyle='',color='red',label='unload')
                    ax.fill_between(np.flip(new_d-new_d[0]),load(new_d),unload(new_d), step="pre", alpha=0.4,color='blue')
                    #plt.text(np.min(new_d), np.max(new_d), r'energy stored = '+str(self.energy_stored[i])+' [mJ]', size=14,ha="left", va="top",bbox=dict(boxstyle="square",ec=(0.5, 0.5, 0.5),fc=(1., 0.8, 0.8)))
                    ax.set_xlabel(r' $d_c$' + str(' ') + ' [mm]',fontsize=18)
                    ax.set_ylabel(r'$F_N$' + str(' ') + ' [N]',fontsize=18)
                    ax.tick_params(bottom=True, top=True, left=True, right=False)
                    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
                    ax.tick_params(axis="x", direction="in",labelsize=18)
                    ax.tick_params(axis="y", direction="in",labelsize=18)
                    ax.tick_params(axis='y', which='minor', direction="in")
                    ax.tick_params(axis='x', which='minor', direction="in")
                    #ax.invert_xaxis()
                    ax.legend(fontsize=14,frameon=False)
                    plt.savefig(self.outfold + '\\plot_of_hysteresys'+ '\\cycle_number_'+str(i+1) +'.png',dpi=200,bbox_inches='tight')
                    plt.savefig(self.outfold + '\\plot_of_hysteresys'+ '\\cycle_number_'+str(i+1) +'.pdf',dpi=200,bbox_inches='tight')
            
        
            else:
                print('no unload curve for the cycle: '+ str(i+1))
                self.energy_stored.append(0)
                
            
            
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
            
        
        if d_0_subtraction == True:
            for i in range(len(self.d)):
                x.append(self.d_0[i]-self.d[i])
            xlabel = r'$d_0$ - $d$' + str(' ') + ' [mm]'
        else:
            for i in range(len(self.d)):
                x.append(self.d[i])
            xlabel = r'$d$' + str(' ') + ' [mm]'
            ax.invert_xaxis()
        
            
        for i in range(len(self.d)):

            ax.plot(x[i],y[i],color=colors[i],label=' load n: = '+ str(i+1) )
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
                ax.plot(x[i],y[i],color=colors[i],label=' load n: = '+ str(i+1))
                ax.set_xlabel(xlabel,fontsize=18)
                ax.set_ylabel(r'$F_N$' + str(' ') + ' [N]',fontsize=18)
                ax.tick_params(bottom=True, top=True, left=True, right=False)
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
                ax.tick_params(axis="x", direction="in",labelsize=18)
                ax.tick_params(axis="y", direction="in",labelsize=18)
                ax.tick_params(axis='y', which='minor', direction="in")
                ax.tick_params(axis='x', which='minor', direction="in")
                
                ax.legend()
                plt.savefig(self.outfold + '\\plot_F_N_vs_d'+ '\\cycle_number_'+str(i+1) +'.png',dpi=200,bbox_inches='tight')
                plt.savefig(self.outfold + '\\plot_F_N_vs_d'+ '\\cycle_number_'+str(i+1) +'.pdf',dpi=200,bbox_inches='tight')
            

        
        
        
        
        return