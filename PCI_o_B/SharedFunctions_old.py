
import numpy as np
import matplotlib.pyplot as plt
import math
from pynverse import inversefunc
from IPython import get_ipython
get_ipython().magic('reset -sf')
import pandas as pd
from scipy.optimize import leastsq, least_squares, curve_fit
import os
from scipy import interpolate
import scipy.integrate as integrate
from scipy.optimize import curve_fit
import re
import pickle as pk
from typing import Union, List, Tuple
import copy



def theta1_func(H_value,R,n1,n2):
    if n1>n2:
        tc=np.arcsin(n2/n1)
        H=lambda theta1 :R*np.sin(theta1)/np.cos(np.arcsin(n1/n2*np.sin(theta1))-theta1)*1/(1-np.tan(np.arcsin(n1/n2*np.sin(theta1))-theta1)/np.tan(np.arcsin(n1/n2*np.sin(theta1))))
        theta=inversefunc(H,y_values=H_value,domain=[-tc, tc])
        
        if H_value>=0:
            h=R*np.sin(theta)/np.cos(np.arcsin(n1/n2*np.sin(theta))-theta)
            theta_scattering=np.arcsin(R*np.sin(theta)/h)
        else:
            h=R*np.sin(theta)/np.cos(np.arcsin(n1/n2*np.sin(theta))-theta)
            theta_scattering=math.pi-np.arcsin(R*np.sin(theta)/h)
    else:
        tc=np.arcsin(n1/n2)
        H=lambda theta1 :R*np.sin(theta1)/np.cos(np.arcsin(n1/n2*np.sin(theta1))-theta1)*1/(1+np.tan(np.arcsin(n1/n2*np.sin(theta1))-theta1)/np.tan(np.arcsin(n1/n2*np.sin(theta1))))
        theta=inversefunc(H,y_values=H_value,domain=[-tc, tc])
        
        if H_value<=0:
            h=R*np.sin(theta)/np.cos(np.arcsin(n1/n2*np.sin(theta))-theta)
            theta_scattering=np.arcsin(R*np.sin(theta)/h)
        else:
            h=R*np.sin(theta)/np.cos(np.arcsin(n1/n2*np.sin(theta))-theta)
            theta_scattering=math.pi-np.arcsin(R*np.sin(theta)/h)
        
    
    return h,theta_scattering


def SingExp(x, amp, decay, baseline ):
    """Model a decaying sine wave and subtract data."""   
    
    model = (amp * np.exp(-x/decay))**2 + baseline
    
    return model

def parab(x, a, b, c):
    return a * x**2 + b * x  + c


def DoubleExp(x, amp1, decay1, amp2, decay2, baseline ):
    """Model a decaying sine wave and subtract data."""
    
    model = (amp1 * np.exp(-x/decay1) + amp2 * np.exp(-x/decay2))**2 + baseline
     
    return model

def DoubleStretchExp(x, amp1, decay1, amp2, decay2, baseline, beta,gamma ):
    """Model a decaying sine wave and subtract data."""
    
    model = (amp1 * np.exp(-x/decay1))**(2*beta) + (amp2 * np.exp(-x/decay2))**(2*gamma) + baseline
     
    return model

def SingleStretchExp(x, amp1, decay1, baseline, beta):
    """Model a decaying sine wave and subtract data."""
    
    model = amp1 * np.exp(-(x/decay1)**beta)   + baseline
     
    return model

def TripleExp(x, amp1, decay1, amp2, decay2, amp3, decay3, baseline ):
    """Model a decaying sine wave and subtract data."""
    
    model = (amp1 * np.exp(-x/decay1) + amp2 * np.exp(-x/decay2) + amp3 * np.exp(-x/decay3) )**2 + baseline
     
    return model

def StretchExp(x, amp, decay, baseline, beta ):
    """Model a decaying sine wave and subtract data."""   
    
    model = (amp * np.exp(-x / decay)**beta) + baseline
    
    return model

def Herzt_Model(d,Force,E,R,nu):
    
    Force = 4/3 * ( E * np.sqrt(R) * d**(3/2) ) / (1 - nu**2)
    
    return Force



def FromTautoTheta(tau,tau_err,T,R_particle,wavelength,nu,n):
    kb=1.38064852*10**-23
    D = kb*T/(6*math.pi*nu*(R_particle*10**-9))
    theta = 2* np.arcsin(  (1/(D*tau))**0.5*wavelength/(4*math.pi*n)  ) *360/(2*math.pi)
    theta_err = 2 * 1 / ( 1- ( wavelength / (4 * n * math.pi * 1 / ( D * tau )**0.5 ) )**2 )**0.5 * wavelength / (8 * n * math.pi) * 1 / D**0.5 * tau**-1.5 * tau_err *360/(2*math.pi)
    return D, theta, theta_err

def SFinterpolation(x,y):
    f = interpolate.interp1d(x, y)
    return f

def SFintegration(x,y,x0,xmax):
    f = interpolate.interp1d(x, y)
    I = integrate.quad(f , x0, xmax)
    return I


def AsymmetryCalculator(func):
    
    ass = []    
    

    for i in range(int(len(func)/2)):
        ass.append( ( np.abs(func[i] - func[-i]) / 2 ) / np.mean(np.asarray(func)) )
        
    
    asymmerty = np.sum(np.asarray(ass))
    
    return asymmerty


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

#################################################################################################


def get_delays(fname):
        """
        given a cI file fname, checkes that the first line corresponds to what
        indeed expected for a cI file (by looking at the name of columns) and 
        returns the time delas as a 1D np.array of int (for cI files with
        delays in number of images) of float (for cI files with delays in s, i.e.
        files with a name that contains '_ts')
        
        Parameters
        ----------
        fname : string
                Name of a cI file
    
        Returns
        -------
        delays : numpy 1D array of int (cI files with delays in # of images) 
                 or float (cI files with delays in s)
                 delays for each column of the cI file, in number of images or s
        """
    
        if '_ts.' in fname:  #cI file with delays in s (e.g. ROI0001_ts.dat)
            dt = np.float
            delay_id = 's'
            init_str = 'tsec n Iav s0.000e+00'
        else :  #cI file with delays in number of images (e.g. ROI0001.dat)
            dt = np.int
            delay_id = 'd'
            init_str = 'n Iav d0'
            
        if not os.path.isfile(fname):
            raise NameError('get_delays(): file\n%s\nnot found' % fname)        
        with open(fname,'r') as f:
            h = f.readline()
        #for simplicity, replace tabs and multiple spaces by simple spaces
        h = " ".join(h.split())
        
        #check if columns are as expected up to first delay column:
        if h[:len(init_str)] != init_str:
            raise NameError('get_delays() err 1 for %s' % fname)
        
        #process all the characters that correspond to cI columns
        index = h.find(delay_id+'0')
        h = h[index:]  #strip all initial characters up to 'd0' or 's0'....
        
        #list of position indexes for 'd' or 's'
        res = [i for i in range(len(h)) if h.startswith(delay_id, i)] 
        
        #check if unexpected characters in the delay col names (anything else 
        #than 'd', ' ', or digits). Only for cI file with delays in number of images 
        if delay_id=='d':
            check = h[res[0]+1:].replace('d','').replace(' ','')
            if  not check.isdigit():
                raise NameError('get_delays() err 2 for %s' % fname)
            
        #get list of delays
        dd = []
        for i in range(len(res)-1):
            dd.append((h[res[i]+1:res[i+1]]))
        dd.append((h[res[-1]+1:]))
        
    
        return np.asarray(dd,dtype=dt) 


def prepare_folder(folder_name):
    """
    creates folder if it does not exist, returns the folder name ending with '/'
    """
    if folder_name[-1] != '\\' and folder_name[-1] != '/': folder_name += '/'
    if not os.path.isdir(folder_name):
        try: 
            os.mkdir(folder_name) 
        except OSError as error: 
            print(error)
            
    return folder_name


def g2_load_pickled(foldername):
    """
    loads and returns a Corr_func object saved as a pickled binary file in 
    folder 'foldername' (filename is standard when saving with Corr_func
    class method g2_save()))
    Use: 
    g2 = g2_load_pickled(foldername), then manipulate g2 as a Corr_func object
    """
    folderout = prepare_folder(foldername)
    with open(folderout+'corr_func.pickle', 'rb') as fin:
        g2 = pk.load(fin)
    return g2


def to_ROI_list(rois: Union[int,List[int]]) -> List[int]:
    """
    given a list of ROIs or a number of ROIs, returns a list of ROI id
    numbers, sorted ascending

    Parameters
    ----------
    rois : Union[int,List[int]]
        DESCRIPTION. if int: number of ROIs to be included in the list 
                     (each element will be set to 1)
                     if list of int: list of ROI id numbers
    Returns
    -------
    List[int]
        DESCRIPTION. Lsit of ROI id numbers, sorted

    """
    if not isinstance(rois,list):  #not a list: create list
        rr = []
        for i in range(int(rois)): rr.append(1)
    else:
        rr = [int(r) for r in rois]
    return sorted(rr)
        

def to_age_list(age: Union[int,List[Tuple[float]]]) -> List[Tuple[float]]:
    """
    given a number of age intervals or a list of ages (list of tuples 
    (tinit,tend) designating time intervals over which cI data are to be 
    averaged to get g2-1) returns a list of ages, sorted ascending

    Parameters
    ----------
    age: Union[int,List[tuple[float]]]
        DESCRIPTION. if int: number of age intervals to be included in the list 
                     (each element will be set to (0,1E6), which in practice
                     means average over the whole set of cI data)
                     if list of tuples: each tuple is like (tinit,tend),
                     tinit and tend in s designate starting/ending time of the 
                     intervals over which cI data are to be averaged
    Returns
    -------
    List[Tuple[float]]
        DESCRIPTION. Lsit of ages, sorted based on tinit values

    """
    if not isinstance(age,list):  #not a list: create list
        aa = []
        for i in range(int(age)): aa.append((0,1E6))
    else:
        aa = age
    return sorted(aa)
        
def build_age_list(tinit,tend,dt,nt,spacing='log',spacing_dt='ratio',\
            mindt=1.):
    """
    builds a list of ages (nt (tstart,tend) tuples denoting time intervals) 
    to be used with a Corr_func object
    Makes sure that the duration of each time interval is at least mindt s 
    """
    
    #check input for consistency
    if tinit>=tend:
        raise NameError('build_age_list(): tinit must be < than tend')
    if dt <= 0:
        raise NameError('build_age_list(): dt must be > 0')
    if spacing =='log' and tinit < 1.:
        raise NameError('build_age_list(): tinit must be >= 1s when' + \
                        ' setting spacing=\'log\'\n')            
    if spacing_dt=='ratio' and dt <=1:
        raise NameError('build_age_list(): dt must be > 1 when' + \
                        ' setting spacing_dt=\'ratio\'')            
    
    
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
    agelist = []
    for iage in range(nt):
        agelist.append((t1[iage],max(t2[iage],t1[iage]+mindt)))
    return agelist
