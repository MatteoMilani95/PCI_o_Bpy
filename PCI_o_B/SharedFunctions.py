
import numpy as np
import matplotlib.pyplot as plt
import math
from pynverse import inversefunc
from IPython import get_ipython
get_ipython().magic('reset -sf')
import pandas as pd
from scipy.optimize import leastsq, least_squares, curve_fit
import os

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


