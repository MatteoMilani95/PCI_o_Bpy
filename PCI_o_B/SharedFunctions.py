
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


def DoubleExp(x, amp1, decay1, amp2, decay2, baseline ):
    """Model a decaying sine wave and subtract data."""
    
    model = (amp1 * np.exp(-x/decay1) + amp2 * np.exp(-x/decay2))**2 + baseline
     
    return model

def DoubleStretchExp(x, amp1, decay1, amp2, decay2, baseline, beta,gamma ):
    """Model a decaying sine wave and subtract data."""
    
    model = (amp1 * np.exp(-x/decay1))**(2*beta) + (amp2 * np.exp(-x/decay2))**(2*gamma) + baseline
     
    return model

def SingleStretchExp(x, amp1, decay1, amp2, decay2, baseline, beta):
    """Model a decaying sine wave and subtract data."""
    
    model = (amp1 * np.exp(-x/decay1))**(2*beta) + (amp2 * np.exp(-x/decay2)) + baseline
     
    return model

def TripleExp(x, amp1, decay1, amp2, decay2, amp3, decay3, baseline ):
    """Model a decaying sine wave and subtract data."""
    
    model = (amp1 * np.exp(-x/decay1) + amp2 * np.exp(-x/decay2) + amp3 * np.exp(-x/decay3) )**2 + baseline
     
    return model

def StretchExp(x, amp, decay, baseline, beta ):
    """Model a decaying sine wave and subtract data."""   
    
    model = (amp * np.exp(-x / decay))**(2*beta) + baseline
    
    return model



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
