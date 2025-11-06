
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
from scipy.signal import savgol_filter
import scipy.optimize, scipy.integrate
from os import walk
import cv2 as cv



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



#################################################  EMA ############################################


def CalcolaVol_MassaEnz (VLudox, phiF):
    
   
    phiI= 0.143
    MuI = 4.66
    Muf = 1
    Enz = 30
   
    
    Vx = (phiI*VLudox)/(phiF)-VLudox
    Vu = (Muf/MuI)*((phiI*VLudox)/(phiF)-phiI*VLudox)
    Ve = Vx-Vu
    me = Enz*(Vx+VLudox-phiI*VLudox)
    
     
    
    str_res  = '\n|---------------|'
    str_res += '\n| Ingredients for a:          '  + str(round(phiF,3)) + ' gel'
    str_res += '\n| Volume of Ludox [mL]:       '  + str(round(VLudox,3))
    str_res += '\n| Volume of Urea  [mL]:       '  + str(round(Vu,3))
    str_res += '\n| Volume of H20 [ml]:         '  + str(round(Ve,3))
    str_res += '\n| Mass of Enzyme [mg]:        '  + str(round(me,3))
    str_res += '\n|--------------------+--------------------|'
    
    print (str_res)
    
    
    return phiF

def DiamToPhi (D_px,scala_px, prima_phi, prima_D_px, scala_px_in):
        
    err_D_px = 5.1
    D_iniz = (0.5*prima_D_px)/scala_px_in
    err_D_iniz = (0.5*err_D_px)/scala_px_in
    D = (0.5*D_px)/scala_px
    phiF = prima_phi*((D_iniz/D))**3
    err_phiF=3*err_D_iniz*prima_phi*(pow(D_iniz, 2)/pow(D, 3))*pow(1+pow(D_iniz, 2)/pow(D, 3), 0.5)
        
        
        
    info = []
    
    info.append(phiF)
    info.append(err_phiF)
    info.append(D)
    info.append(err_D_iniz)
    #info.append(D_iniz)
    #info.append(err_D_iniz)
        
    return info


def load_results_from_SLoad(path):
    a = pd.read_csv(path, index_col=None,skiprows=1,names= ['phi','d 0','Young Modulus','indirizzo'],usecols=[1,2,3,4],sep='\,', decimal=".")
    
    return a

def fancy_plots_phi_young(phi,YM,color_plot,name,symbol):
    
    
    
    
    plt.figure()
    ax = plt.axes()
    ax.loglog(phi,YM,marker = symbol, linestyle='',label=name,color=color_plot,markersize=10)
    ax.set_xlabel(r'$\phi$ ' + str(' ') + ' [-]',fontsize=18)
    ax.set_ylabel(r'$E$' + str(' ') + ' [Pa]',fontsize=18)
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.tick_params(axis="x", direction="in",labelsize=18)
    ax.tick_params(axis="y", direction="in",labelsize=18)
    ax.tick_params(axis='y', which='minor', direction="in")
    ax.tick_params(axis='x', which='minor', direction="in")
    ax.legend()
    
    
    return

def fit_curver_line_loglog(PHI,YOUNG_MODULI,ax,alpha_x,alpha_y):
    
    def line(x,alpha,C):
        y = x*alpha+np.log(C) 
        return y
    def power_law(x,alpha,C):
        y = C*x**alpha 
        return y
    
    popt, pcov = curve_fit(line, np.log(PHI),np.log(YOUNG_MODULI))
    alpha = popt[0]
    C = popt[1]
    print(popt)
    
    fake_phi = np.linspace(0.01,0.5,100)
    
    
    
    ax.set_yscale("log")
    ax.set_xscale("log")
    
    ax.plot(fake_phi, power_law(fake_phi,alpha,C),color='red',linewidth=3)
    ax.set_xlabel(r'$\varphi$ ' + str(' ') + ' [-]',fontsize=15)
    ax.set_ylabel(r'$E$' + str(' ') + ' [Pa]',fontsize=15)
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.tick_params(axis="x", direction="in",labelsize=18)
    ax.tick_params(axis="y", direction="in",labelsize=18)
    ax.tick_params(axis='y', which='minor', direction="in")
    ax.tick_params(axis='x', which='minor', direction="in")
    
    print ('write the plot you want to put in ')
    
    ax.text(alpha_x, alpha_y, r'$\alpha$ = '+str( np.round(alpha,2)), size=14,ha="left", va="top",bbox=dict(boxstyle="square",ec=(0.5, 0.5, 0.5),fc=(1., 0.8, 0.8)))
    

    
def load_results_drying(path):
    a = pd.read_csv(path, index_col=None,skiprows=1,names= ['time','diameter','diameter error','R/R0 [-]','phi [-]','err_phi[-]','Young Moduli [Pa]',],usecols=[0,1,2,3,4,5,6],sep='\,', decimal=".")
    
    return a

'''def load_results_compression(path):
    a = pd.read_csv(path, index_col=None,skiprows=1,names= ['time','diameter','diameter error','R/R0 [-]','phi [-],'Young Moduli [Pa]',],usecols=[0,1,2,3,4,5,6,7],sep='\,', decimal=".")
    
    return a
'''
def smooth_data(x,y,a,b,ax):
    yhat = savgol_filter(y, a, b)
    ax.plot(x,yhat, color= 'blue')
    
def excess_xydata_average (x, y):
    counter = 0.
    sumy = 0.
    new_y = []
    new_x = []
    new_z = []
    j = 0
    d = len(x)
  
    
    while (j<d-1):
        while (x[j]==x[j+1]):
            sumy = sumy + y[j]
            counter = counter + 1.
            j=j+1
            
            if (j == d-1):
                break
            if (x[j]!=x[j+1]):
                sumy = sumy + y[j]
                new_x.append(x[j])
                counter = counter + 1.
                new_y.append(sumy/counter)
                j=j+1
                counter = 0.
                sumy = 0.
                if (j == d-1):
                    break
        
        counter = 0.
        sumy = 0.
        new_x.append(x[j])
        new_y.append(y[j])
        j= j+1
    
    new_y.append(y[d-1])
    new_x.append(x[d-1])
    
    
    new_yarray = np.array(new_y)
    new_xarray = np.array(new_x)        
    return new_xarray,new_yarray

        

def swap (x,y):
    t = x
    x = y
    y = t    
    

def area_Hertz (Forza, E_Hertz, d0):
    a = []
    for i in range (len(Forza)):
        a.append(3.14*pow((3*Forza[i]*d0[i])/(8*E_Hertz[i]),2/3))
    a_array = np.array(a)
    
    return a_array
   
    
def load_results_from_SLoad_ALL(path):
    a = pd.read_csv(path, index_col=None,skiprows=1,names= ['number','d0 - d', 'Normal Force','phi', 'contact points','engineering_stress' ,'strain','time'],usecols=[1,2,3,4,5,6,7],sep='\,', decimal=".",engine='python')
          
    return a   

def excess_xydata_average_skip_interval (x, y,a,b):
    counter = 0.
    sumy = 0.
    new_y = []
    new_x = []
    c = len(x)
  
    j = 0
    
  
    while (j<c-1):
        
        while (x[j]==x[j+1]):
            sumy = sumy + y[j]
            counter = counter + 1.
            j=j+1
            if(j==a):
                j=b
                
                for i in range(a,b,1):
                    new_y.append(y[i])
                    new_x.append(x[i])
            
            if (j == c-1):
                break
            if (x[j]!=x[j+1]):
                sumy = sumy + y[j]
                new_x.append(x[j])
                counter = counter + 1.
                new_y.append(sumy/counter)
                j=j+1
                if(j==a):
                    j=b
                    for i in range(a,b,1):
                        
                        new_y.append(y[i])
                        new_x.append(x[i])
                counter = 0.
                sumy = 0.
                if (j == c-1):
                    break
        
        counter = 0.
        sumy = 0.
        new_x.append(x[j])
        new_y.append(y[j])
        j= j+1
        if(j==a):
            j=b
            
            for i in range(a,b,1):
                new_y.append(y[i])
                new_x.append(x[i])   
                
            
    
    new_y.append(y[c-1])
    new_x.append(x[c-1])
    
    new_yarray = np.array(new_y)
    new_xarray = np.array(new_x)        
    return new_xarray,new_yarray
        


    
def load_results_from_SLoad_experiment_info(path):
    a = pd.read_csv(path, index_col=None,skiprows=1,names= ['d0 - d', 'Normal Force','phi', 'contact points','Hertz engineering_stress','strain'],usecols=[1,2,3,4,5,6],sep='\,', decimal=".")
          
    return a       
     
def load_results_from_SLoad_general_information(path):
    a = pd.read_csv(path, index_col=None,skiprows=1,names= ['phi [-]', 'strain rate [1/s]','d_0 [mm]', 'costant Hertz [Pa]','E [Pa]','poisson ratio [-]','yealding strain [-]','yealding force [-]','Delta Force[-]', 'Delta Force su Delta strain [-]','theta [rad]'],usecols=[1,2,3,4,5,6,7,8,9,10,11],sep='\,', decimal=".")
          
    return a    

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
     

def power_law(x, m, q):
    return q*x**m



def particle_size(q,a,R_0,sigma):
    
    
    
    func = lambda R : a * np.exp( -(R-R_0)**2 / (2*sigma**2) ) *( 3 * (np.sin(q*R) - q*R*np.cos(q*R))*(q*R)**-3 )**2
    
    model = scipy.integrate.quad(func , -np.inf, np.inf)
    
    return model[0]

def line(x,a,b):
    return  a*x +b

def fit_hertz(d_array,F_n_array,R):
    
    Force_Hertz = lambda d , E , C:  4/3 *  E * np.sqrt(R)   * (d)**(3/2)   + C
    
    popt, pcov = curve_fit(Force_Hertz, d_array , F_n_array)    #,bounds=([1e-10,-10],[1e9,10])
    E = popt[0]
    C = popt[1]
    
  
    return E,C


def fit_hertz_2_0(d_array,F_n_array,R):
    
    Force_Hertz = lambda d , E , C,d_0:  4./3. *  E * np.sqrt(R)   * (np.abs(d-d_0))**(3/2)  +C
    
    popt, pcov = curve_fit(Force_Hertz, d_array , F_n_array,bounds=([1e-10,-1,0.82],[2000,1,0.84]))    #,bounds=([1e-10,-10],[1e9,10])
    E = popt[0]
    C = popt[1]
    d_0 = popt[2]
 
    
  
    return E,C,d_0




def load_shear_rheology(path):
    a = pd.read_csv(path, index_col=None,skiprows=[0,1,2,3,4,5,6,7,8,9],names= ['strain','stress','Gprime','Gdoubleprime'],usecols=[2,3,4,5],sep='\t', decimal=",",encoding='UTF-16 LE')
          
    return a 

def Fisher_Burford(x,A,df,R_c):
    
    model = np.log(A / np.power( 1+ x**2 * R_c**2 / 3*(df/2),df/2))
    
    return model


def Guinier_lin_log(x,ln_I0,Rg):
    
    model = ln_I0 - (x**2 * Rg**2 / 3)
    
    return model

def CheckFileExists(filePath):
    try:
        return os.path.isfile(filePath)
    except:
        return False

def CheckFolderExists(folderPath):
    return os.path.isdir(folderPath)

def CheckCreateFolder(folderPath):
    if (os.path.isdir(folderPath)):
        return True
    else:
        print("Created folder:" + folderPath)
        os.makedirs(folderPath)
        return False

def FindFileNames(FolderPath):
    FilenameList = []
    # get list of all filenames in raw difference folder
    for (dirpath, dirnames, filenames) in walk(FolderPath):
        FilenameList.extend(filenames)
        break
    return FilenameList

def find_closest_index(arr, threshold):
    """
    Finds the index of the value in the array that is closest to the given threshold,
    ignoring NaN values.

    Parameters:
    arr (list or array-like): The array to search.
    threshold (float or int): The threshold value to compare against.

    Returns:
    int: The index of the value closest to the threshold. Returns None if all values are NaN.
    """
    

    # Convert the array to a NumPy array
    arr_np = np.array(arr)

    # Check for all NaN array
    if np.all(np.isnan(arr_np)):
        return None

    # Calculate absolute differences, ignoring NaN values
    diff = np.abs(arr_np - threshold)

    # Replace NaN values in diff with infinity to ignore them
    diff[np.isnan(diff)] = np.inf

    # Find the index of the minimum difference
    closest_index = np.argmin(diff)

    return closest_index

def find_local_minima(arr):
    # Ensure the input is a numpy array
    arr = np.asarray(arr)
    
    # Create a boolean array where True indicates a local minimum
    local_minima = (np.r_[True, arr[1:] < arr[:-1]] & np.r_[arr[:-1] < arr[1:], True])
    
    # Get the indices of the local minima
    local_minima_indices = np.where(local_minima)[0]
    
    return local_minima_indices

def list_filenames(folder_path,spacing):
    try:
        # Get a list of all files and directories in the specified folder
        with os.scandir(folder_path) as entries:
            filenames = [entry.name for idx, entry in enumerate(entries) if entry.is_file() and idx % spacing == 0]
        return filenames
    except FileNotFoundError:
        print("The specified folder does not exist.")
        return []
    
    
def first_below_threshold_index(arr, threshold):
    try:
        return np.where(arr < threshold)[0][0]
    except IndexError:
        return -1 


import pandas as pd

def load_data(path, skip_lines):
    """
    Load columns 3 (time), 4 (normal force) and 7 (gap) from a tab-separated file,
    skipping the first `skip_lines` lines, and stop as soon as the 2nd column is empty.

    Parameters
    ----------
    path : str
        Path to the data file.
    skip_lines : int
        Number of lines to skip at the top of the file.

    Returns
    -------
    time : np.ndarray
        Time values (column 3).
    force : np.ndarray
        Normal force values (column 4).
    gap : np.ndarray
        Gap values (column 7).
    nrows : int
        Number of rows actually loaded.
    """
    # We use usecols to pull in columns [1,2,3,6] (0-based)
    df = pd.read_csv(
        path,
        encoding = "utf-16",
        sep='\t',
        header=None,
        skiprows=skip_lines,
        usecols=[1, 2, 3, 6],
        names=['col2', 'time', 'force', 'gap'],
        dtype={'col2': str}  # read the 2nd column as string so empty cells become NaN
    )

    # Identify the first row where col2 is missing or blank
    if df['time'].isna().any():
       first_nan_idx = df['time'].isna().idxmax()
       df = df.iloc[:first_nan_idx]

    # Extract and return
    time  = df['time'].to_numpy(dtype=float)
    force = df['force'].to_numpy(dtype=float)
    gap   = df['gap'].to_numpy(dtype=float)
    nrows = len(df)

    return time, force, gap, nrows



def tamplate_match_box(img,template):
    
    
    res = cv.matchTemplate(img, template, 3)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    
    h, w = template.shape[:2] 
    
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)  # Moved this line up before using it
    
    bbox  = (top_left[0],top_left[0]+w,top_left[1],top_left[1]+h)
    return bbox

           
def list_txt_files(folder_path):
    txt_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                full_path = os.path.join(root, file)
                txt_files.append(full_path)
    return txt_files


def trim_if_odd(lst):
    if len(lst) % 2 == 1:  # Check if length is odd
        lst = lst[:-1]     # Remove the last element
    return lst
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
