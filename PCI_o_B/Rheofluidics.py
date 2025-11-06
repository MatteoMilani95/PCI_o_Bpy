# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 13:28:32 2025

@author: matte
"""

from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from glob import glob
import os
import matplotlib.pylab as pl
from skimage import img_as_ubyte
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter, median_filter
from skimage.feature import peak_local_max
from scipy.interpolate import interp1d
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from scipy.ndimage import label
from skimage.feature import blob_log
from skimage import io
from skimage.measure import find_contours, regionprops
from matplotlib.path import Path
from skimage.measure import EllipseModel
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Callable, Iterable, Any, List, Tuple
import gc
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature,io, img_as_float, exposure, restoration, filters, util
from scipy.fft import fft, ifft, fftfreq
import itertools
from scipy.optimize import curve_fit
import pandas as pd
from pathlib import Path
from matplotlib.path import Path as MplPath



def cos_fourier(theta, *a):
    result = a[0]
    for n in range(1, len(a)):
        result += a[n] * np.cos(n * theta)
    return result

def tamplate_match_box(img,template):
    
    
    res = cv2.matchTemplate(img, template, 3)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    h, w = template.shape[:2] 
    
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)  # Moved this line up before using it
    
    bbox  = (top_left[0],top_left[0]+w,top_left[1],top_left[1]+h)
    return bbox

def crop_image(img, box):
        x0, x1, y0, y1 = box
        cropped = img[y0:y1, x0:x1]
        del img  # free original
        gc.collect()
        return cropped
    
def stabilize_and_crop_border(image, x,y, border=4):
    """
    Shift `image` so that its center-of-mass `com` ends up at the center,
    then crop `border` pixels off each side.

    :param image:  input image as a NumPy array
    :param com:    (x, y) tuple of the object's center of mass in `image`
    :param border: number of pixels to trim off each side after shifting
    :returns:      shifted & cropped image
    """
    h, w = image.shape[:2]

    # 1) Compute translation so com -> center of original image
    x_ref, y_ref = w / 2.0, h / 2.0
    dx = x_ref - y
    dy = y_ref - x
    M = np.array([[1, 0, dx],
                  [0, 1, dy]], dtype=np.float32)

    # 2) Warp the full image
    shifted_full = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # 3) Crop off `border` pixels on all sides
    cropped = shifted_full[border : h - border,
                            border : w - border]

    return cropped

    
def edge_with_Canny(img,sig):
    
    if len(img.shape) == 3:
    
        image = img[:,:,0]
    else:
        image = img

    # 2. Apply Canny with two sigma values
    edges2 = feature.canny(image, sigma=5)
    
    ys, xs = np.where(edges2 != 0)
    return ys, xs

def fit_Fourier_cos_only(x,y,N):
    

    # 1) Choose a center (cx, cy) around which to measure r,θ
    #    For closed shapes, the centroid is a good choice:
    cx = np.mean(x)
    cy = np.mean(y)
    
    # 2) Compute polar coords relative to center
    dx = x - cx
    dy = y - cy
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)             # in [-π,π]
    theta = np.mod(theta, 2*np.pi)         # map to [0,2π)
    
    # 3) Sort points by θ
    order = np.argsort(theta)
    theta_u = theta[order]
    r_u     = r[order]
    # 3) Initial guess: a0 = mean radius, others = 0
    p0 = np.zeros(N+1)
    p0[0] = np.mean(r_u)
    
    # 4) Fit the coefficients a0…a4
    popt, pcov = curve_fit(cos_fourier, theta_u, r_u, p0=p0)
    
    # 5) Reconstruct on fine grid
    theta_fit = np.linspace(0, 2*np.pi, 1000)
    r_fit     = cos_fourier(theta_fit, *popt)
    return popt, pcov,theta_fit,r_fit,theta_u,r_u

def load_PIV_from_Matlab(path):
    df = pd.read_csv(path, skiprows=4, header=None)
    
    # Assign columns to variables
    x = df[0]
    y = df[1]
    v_x = df[2]
    v_y = df[3]
    
    x = df[0].values
    y = df[1].values
    v_x = df[2].values
    v_y = df[3].values
    
    return x,y,v_x,v_y

def refine_velocities_PIV(x,y,v_x,v_y,x_edges,y_edges):
    
    xc, yc = np.mean(x_edges), np.mean(y_edges)

    # 2) Compute angles of each edge point about that center:
    angles = np.arctan2(y_edges - yc, x_edges - xc)
    
    # 3) Sort the edge points by angle so they form a proper loop:
    order = np.argsort(angles)
    verts = np.column_stack((x_edges[order], y_edges[order]))
    
    # 4) Close the polygon:
    if not np.allclose(verts[0], verts[-1]):
        verts = np.vstack((verts, verts[0]))
    
    # 5) Build the Path and test points:
    poly = MplPath(verts)
    points = np.column_stack((x, y))
    inside = poly.contains_points(points)   # boolean mask
    '''
    # 6) Zero out velocities outside:
    if diffsx[index] > 0:
        v_x_masked = v_x.copy()#-diffsx[index]/8
    else:
        v_x_masked = v_x.copy()#+diffsx[index]
    
    if diffsy[index] > 0:
        v_y_masked = v_y.copy()#-diffsy[index]/8
    else:
        v_y_masked = v_y.copy()#+diffsy[index]
    print(diffsx[index])
    print(diffsy[index])
        
    '''
    v_x_masked = v_x.copy()
    v_y_masked = v_y.copy()
    v_x_masked[~inside] = np.nan
    v_y_masked[~inside] = np.nan
    return v_x_masked,v_y_masked,verts

def refine_contrast(img, contrast, brightness):
    
    contrast = contrast  # contrast
    brightness = brightness     # brightness

    adjusted = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64,64))
    clahe_u8 = (img*255).astype('uint8')
    clahe_res = clahe.apply(clahe_u8) / 255.0
    
    return clahe_res

def non_local_means(img):
    if img.ndim == 3:
        img_gray = cv2.cvtColor((img*255).astype('uint8'), cv2.COLOR_RGB2GRAY) / 255.0
    else:
        img_gray = img.copy()
    
    # NLM denoise with channel_axis correction
    patch_kw = dict(patch_size=5, patch_distance=6)
    nlm = restoration.denoise_nl_means(img_gray, 
                                       h=0.8*np.std(img_gray), 
                                       fast_mode=True, 
                                       channel_axis=None, 
                                       **patch_kw)
    out_img = cv2.merge([nlm, nlm, nlm])
    return out_img

def images_for_PIV(img):
    if img.ndim == 3:
        img_gray = cv2.cvtColor((img*255).astype('uint8'), cv2.COLOR_RGB2GRAY) / 255.0
    else:
        img_gray = img.copy()
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64,64))
    clahe_u8 = (img_gray*255).astype('uint8')
    clahe_res = clahe.apply(clahe_u8) / 255.0
    
    # Background subtraction
    large_sigma = 25
    bg = filters.gaussian(img_gray, sigma=large_sigma)
    bg_sub = img_gray - bg
    bg_sub = np.clip(bg_sub, 0, 1)
    
    # Bandpass (DoG)
    low_sigma = 1.0
    high_sigma = 3.0
    dog = filters.gaussian(img_gray, low_sigma) - filters.gaussian(img_gray, high_sigma)
    dog = (dog - dog.min()) / (dog.max() - dog.min() + 1e-12)
    
    # NLM denoise with channel_axis correction
    patch_kw = dict(patch_size=5, patch_distance=6)
    nlm = restoration.denoise_nl_means(img_gray, h=0.8*np.std(img_gray), fast_mode=True, channel_axis=None, **patch_kw)
    
    # Combined pipeline
    combined = clahe_res.copy()
    combined = combined - filters.gaussian(combined, sigma=large_sigma)
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-12)
    combined = filters.gaussian(combined, sigma=0.8)
    combined = restoration.denoise_nl_means(combined, h=0.6*np.std(combined), fast_mode=True, channel_axis=None, **patch_kw)
    combined = np.clip(combined, 0, 1)
    
    out_img = cv2.merge([combined, combined, combined])
    return out_img

def find_and_crop_droplets_in_list(
        imgs,
        x_top=500, y_top=150, w=150, h=120,
        threshold=0.6, overlap_thresh=0.3, y_shift=-15,
        save_dir=None
    ):
    """
    Detect and crop droplets in a list of images.
    Works with grayscale or RGB images.
    Cropped droplets are saved with progressive numbering across all frames.
    """

    # === Simple NMS ===
    def nms(boxes, overlapThresh=0.3):
        if len(boxes) == 0:
            return []
        boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes])
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        pick = []
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            if last == 0:
                break
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w_ = np.maximum(0, xx2 - xx1 + 1)
            h_ = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w_ * h_) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        return boxes[pick].astype("int")

    # === Prepare template from first image ===
    img0 = imgs[0]
    if img0.ndim == 3:
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    template = img0[y_top:y_top+h, x_top:x_top+w]
    H, W = img0.shape

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    crops_all, ids_all, xs_all, ys_all, frames_all, paths_all = [], [], [], [], [], []

    drop_idx = 1  # progressive counter

    # === Process each image ===
    for frame_idx, img in enumerate(imgs):
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        detections = [[int(x), int(y), w, h] for x, y in zip(loc[1], loc[0])]
        filtered_boxes = nms(detections, overlap_thresh)

        half_w, half_h = w // 2, h // 2
        for (x1, y1, x2, y2) in filtered_boxes:
            cx = x1 + (x2 - x1) // 2
            cy = y1 + (y2 - y1) // 2
            x_start = max(0, cx - half_w)
            y_start = max(0, cy - half_h + y_shift)
            x_end = min(W, x_start + w)
            y_end = min(H, y_start + h)
            crop = img[y_start:y_end, x_start:x_end]

            # save
            if save_dir:
                filename = f"drop_{drop_idx:04d}.tif"
                full_path = os.path.join(save_dir, filename)
                cv2.imwrite(full_path, crop)
                paths_all.append(full_path)
            else:
                paths_all.append(None)

            crops_all.append(crop)
            ids_all.append(drop_idx)
            xs_all.append(x_start)
            ys_all.append(y_start)
            frames_all.append(frame_idx + 1)

            drop_idx += 1

    # === Save CSV ===
    if save_dir:
        csv_path = os.path.join(save_dir, "droplet_coordinates.csv")
        pd.DataFrame({
            "frame": frames_all,
            "id": ids_all,
            "x": xs_all,
            "y": ys_all,
            "path": paths_all
        }).to_csv(csv_path, index=False)
        print(f"✅ Saved coordinates CSV: {csv_path}")

    return crops_all, ids_all, xs_all, ys_all


def moving_average(y, window_size=5):
    return np.convolve(y, np.ones(window_size)/window_size, mode='same')
from scipy.signal import savgol_filter

def determine_distances(path):
    df = pd.read_csv(path)  # replace with your CSV file path

    # Step 2: Select the last two columns
    last_two_cols = df.iloc[:, -2:]  # iloc selects columns by index
    
    # Step 3: Convert them to numpy arrays
    array1 = last_two_cols.iloc[:, 1].to_numpy()
    array2 = last_two_cols.iloc[:, 0].to_numpy()
    
    # Optional: combine into a single 2D numpy array
    combined_array = last_two_cols.to_numpy()
    
    indices = np.where(array1 > np.mean(array1))[0]
    up_y = array1[indices]
    up_x = array2[indices]
    
    indices = np.where(array1 < np.mean(array1))[0]
    down_y = array1[indices]
    down_x = array2[indices]

    x=up_x
    y=up_y
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Create interpolation function with extrapolation
    interp_func = interp1d(x_sorted, y_sorted, kind='linear', fill_value="extrapolate")
    
    # Define new x-values, extending beyond the original boundaries
    x_new = np.linspace(0, 1000, 500)  # example: 5 units beyond min/max
    y_new = interp_func(x_new)

    x=down_x
    y=down_y
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Create interpolation function with extrapolation
    interp_func = interp1d(x_sorted, y_sorted, kind='linear', fill_value="extrapolate")
    
    # Define new x-values, extending beyond the original boundaries
    x_new = np.linspace(0, 1000, 500)  # example: 5 units beyond min/max
    y_new2 = interp_func(x_new)

    distances = y_new2 - y_new

    distances_smooth = moving_average(distances , window_size=15)
    distances_smooth = savgol_filter(distances, window_length=21, polyorder=3) 
    
    
    
    
   


    
    return x_new,distances_smooth


def calc_stress_middle_channel(x,distances,visc=20,avg_q=1):

    
    
    spl_xarr = x #position where channel thickness is evauated (vector)
    
    
    ch_thick = distances#(vector)
    ch_th_grad = np.gradient(ch_thick, spl_xarr*1e-6) #unitless
    ch_ext_stress = avg_q * visc * ch_th_grad / np.square(ch_thick)

    return spl_xarr,ch_ext_stress


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# --- Sine model ---
def sine_model(x, A, f, phi, offset):
    return A * np.sin(2 * np.pi * f * x + phi) + offset

# --- Main function with adjustable number of fit points ---
def fit_sine(x, y, n_points=2000, plot=True):
    """
    Fit a sine wave to data (x, y).

    Parameters
    ----------
    x : array-like
        Independent variable
    y : array-like
        Dependent variable
    n_points : int, optional
        Number of points in x_fit and y_fit. Default is 2000.
    plot : bool, optional
        If True, plots data, fit, and residuals. Default is True.

    Returns
    -------
    x_fit : np.ndarray
        Fine x grid for plotting the fitted sine
    y_fit : np.ndarray
        Fitted sine values on x_fit
    popt : array
        Optimized parameters [A, f, phi, offset]
    r_squared : float
        Coefficient of determination of the fit
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # --- Initial guesses ---
    A_guess = 0.5 * (np.nanmax(y) - np.nanmin(y))
    offset_guess = np.nanmean(y)
    
    # Estimate frequency via FFT
    def estimate_freq_fft(x, y, oversample=4):
        N_uniform = int(len(x) * oversample)
        if N_uniform < 8:
            return 0.0
        f_interp = interp1d(x, y - np.mean(y), kind='linear', fill_value="extrapolate")
        x_uniform = np.linspace(x.min(), x.max(), N_uniform)
        y_uniform = f_interp(x_uniform)
        Y = np.fft.rfft(y_uniform)
        freqs = np.fft.rfftfreq(N_uniform, d=(x_uniform[1] - x_uniform[0]))
        power = np.abs(Y)
        power[0] = 0.0
        peak_idx = np.argmax(power)
        return float(freqs[peak_idx])

    f_guess = estimate_freq_fft(x, y, oversample=6)
    if f_guess <= 0:
        f_guess = 1.0 / (x.max() - x.min() + 1e-12)
    
    phi_guess = 0.0
    p0 = [A_guess, f_guess, phi_guess, offset_guess]

    bounds = (
        [0.0, 0.0, -2*np.pi, -np.inf],
        [2*(np.nanmax(y)-np.nanmin(y)), np.inf, 2*np.pi, np.inf]
    )

    # Fit sine curve
    try:
        popt, pcov = curve_fit(sine_model, x, y, p0=p0, bounds=bounds, maxfev=20000)
    except RuntimeError:
        popt, pcov = curve_fit(sine_model, x, y, p0=p0, maxfev=20000)

    # Predicted values and residuals
    y_pred = sine_model(x, *popt)
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Fine grid for plotting (with user-controlled number of points)
    x_fit = np.linspace(x.min(), x.max(), n_points)
    y_fit = sine_model(x_fit, *popt)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,6), gridspec_kw={'height_ratios':[3,1]}, sharex=True)
        ax1.scatter(x, y, s=20, label='Data', alpha=0.6)
        ax1.plot(x_fit, y_fit, label='Sine fit', linewidth=2)
        ax1.set_ylabel('y')
        ax1.legend()
        ax1.set_title('Sine fit to data')

        ax2.plot(x, residuals, 'o', markersize=4, alpha=0.7)
        ax2.axhline(0, linestyle='--')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Residuals')
        plt.tight_layout()
        plt.show()

    return x_fit, y_fit, popt, r_squared

def compute_dx(df, max_match_dist=100):
    """
    Compute dx (change in x between consecutive frames)
    for each droplet by matching detections across frames.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ['frame', 'x'].
        Each row is a droplet detection at a given frame.
    max_match_dist : float, optional
        Maximum allowed displacement (in pixels) to consider
        two detections as belonging to the same droplet.

    Returns
    -------
    dx : np.ndarray
        Array of dx values, same length and order as df.
    """
    # Ensure clean order
    df = df.reset_index(drop=True)
    frames = sorted(df['frame'].unique())
    frame_to_indices = {f: df.index[df['frame'] == f].tolist() for f in frames}
    dx = np.zeros(len(df), dtype=float)

    # Active droplet tracks (each track = dict)
    tracks = []

    for i, f in enumerate(frames):
        indices = frame_to_indices[f]
        xs = df.loc[indices, 'x'].values.astype(float)

        # First frame: initialize tracks
        if i == 0:
            for idx, x in zip(indices, xs):
                tracks.append({'last_x': x, 'last_idx': idx})
                dx[idx] = 0.0
            continue

        # Build cost matrix for greedy matching
        T, D = len(tracks), len(xs)
        if T == 0 or D == 0:
            for idx, x in zip(indices, xs):
                tracks.append({'last_x': x, 'last_idx': idx})
                dx[idx] = 0.0
            continue

        cost = np.abs(np.array([tr['last_x'] for tr in tracks])[:, None] - xs[None, :])
        cost_copy = cost.copy()
        matched_tracks, matched_dets, assignments = set(), set(), []

        while True:
            ti, di = np.unravel_index(np.argmin(cost_copy), cost_copy.shape)
            minval = cost_copy[ti, di]
            if minval > max_match_dist or np.isinf(minval):
                break
            if ti in matched_tracks or di in matched_dets:
                cost_copy[ti, di] = np.inf
                if np.isinf(cost_copy).all():
                    break
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            assignments.append((ti, di))
            cost_copy[ti, :] = np.inf
            cost_copy[:, di] = np.inf
            if np.isinf(cost_copy).all():
                break

        matched_det_indices = set()
        for ti, di in assignments:
            tr = tracks[ti]
            det_idx = indices[di]
            x_cur = xs[di]
            dx[det_idx] = x_cur - tr['last_x']
            tr['last_x'] = x_cur
            tr['last_idx'] = det_idx
            matched_det_indices.add(di)

        # New detections → start new tracks
        for j, (idx, x) in enumerate(zip(indices, xs)):
            if j not in matched_det_indices:
                tracks.append({'last_x': x, 'last_idx': idx})
                dx[idx] = 0.0

    return dx

def remove_outliers(reference, *arrays, severity=1.5):
    """
    Removes outliers based on IQR with adjustable severity.
    
    severity < 1.5 → more aggressive filtering
    severity > 1.5 → more tolerant
    """
    ref_array = np.asarray(reference)
    
    Q1 = np.percentile(ref_array, 25)
    Q3 = np.percentile(ref_array, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - severity * IQR
    upper_bound = Q3 + severity * IQR
    
    mask = (ref_array >= lower_bound) & (ref_array <= upper_bound)
    
    filtered_arrays = []
    for arr in (reference, *arrays):
        if isinstance(arr, (pd.DataFrame, pd.Series)):
            filtered_arrays.append(arr[mask].copy())
        else:
            filtered_arrays.append(np.asarray(arr)[mask])
    
    return tuple(filtered_arrays)






