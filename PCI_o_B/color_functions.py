# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:51:16 2024

@author: hp
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
import pandas as pd
from numpy import sin, cos
from numpy.linalg import lstsq
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure
from skimage import filters, feature, color, data
from scipy.ndimage import label, generate_binary_structure

import itertools



def load_images_from_folder(folder,extesion):
    image_paths = glob(os.path.join(folder, "*" +extesion))  # Adjust to match the image format (e.g., .jpg, .jpeg, etc.)
    images = [cv2.imread(img_path) for img_path in image_paths]
    
    return images


def load_images_from_folder_2_0(folder, extension, skip=0, stop=None, step=1):
    """
    Load images with the specified extension from a folder, optionally skipping, stopping,
    and picking images at a defined interval (step).

    Parameters:
    - folder (str): The folder path containing images.
    - extension (str): The file extension to filter images (e.g., '.jpg', '.png').
    - skip (int): The number of initial images to skip. Default is 0.
    - stop (int or None): The index at which to stop loading images. If None, loads until the end.
    - step (int): The interval at which to pick images (e.g., step=10 picks every 10th image). Default is 1.

    Returns:
    - list: A list of loaded images.
    """
    # Ensure the extension starts with a dot
    if not extension.startswith("."):
        extension = "." + extension

    # Get all image file paths with the given extension
    image_paths = sorted(glob(os.path.join(folder, "*" + extension)))  # Sort to maintain order
    if not image_paths:
        print(f"No images found in folder '{folder}' with extension '{extension}'")
        return []

    # Apply skip, stop, and step
    image_paths = image_paths[skip:stop:step]

    # Read each image, filtering out any that fail to load
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Failed to load image at {img_path}")

    print(f"Loaded {len(images)} images from '{folder}' (skip={skip}, stop={stop}, step={step})")
    return images

def _apply_args(func: Callable, args: Tuple) -> Any:
    """
    Unpack args tuple and call func.
    """
    return func(*args)


def _read_image(path: str) -> Any:
    """
    Read an image from disk, returning None on failure.
    """
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Failed to load image at {path}")
    return img


def run_in_parallel(
    func: Callable,
    args_list: Iterable[Tuple],
    num_workers: int = None,
    desc: str = "Processing",
    unit: str = "item"
) -> List[Any]:
    """
    Run a function in parallel over a list of argument tuples, showing a progress bar.
    Results are returned in the same order as args_list.

    Parameters:
    - func: function to execute, accepting positional args matching each tuple in args_list.
    - args_list: iterable of tuples to pass to func.
    - num_workers: number of processes. Defaults to os.cpu_count().
    - desc: description for tqdm progress bar.
    - unit: unit name for tqdm.

    Returns:
    - List of func results, in original order.
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 1

    results: List[Any] = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        mapped = executor.map(
            _apply_args,
            itertools.repeat(func),
            args_list
        )
        for res in tqdm(mapped, total=len(args_list), desc=desc, unit=unit):
            results.append(res)
    return results


def load_images_from_folder_parallel(
    folder: str,
    extension: str,
    skip: int = 0,
    stop: int = None,
    step: int = 1,
    num_workers: int = None
) -> List[Any]:
    """
    Load images with the specified extension from a folder in parallel,
    preserving the original order.
    """
    if not extension.startswith('.'):
        extension = f".{extension}"
    pattern = os.path.join(folder, f"*{extension}")
    paths = sorted(glob(pattern))[skip:stop:step]
    if not paths:
        print(f"No images found in '{folder}' with extension '{extension}'")
        return []

    args_list = [(p,) for p in paths]
    images = run_in_parallel(
        func=_read_image,
        args_list=args_list,
        num_workers=num_workers,
        desc="Loading images",
        unit="img"
    )
    images = [img for img in images if img is not None]
    print(f"Loaded {len(images)} images (skip={skip}, stop={stop}, step={step})")
    return images



def adjust_hue(img, shift):
        return (img[:, :, 0] + shift) % 180

def crop_image(img, box):
        x0, x1, y0, y1 = box
        cropped = img[y0:y1, x0:x1]
        del img  # free original
        gc.collect()
        return cropped
    
    
def whitepatch(image, x_top, y_top, 
                         w, h):
    image_patch = image[y_top:y_top+h, 
                        x_top:x_top+w]

    return image_patch

def white_correction(image,image_patch):
    
    image_max = (image*1.0 / 
                 image_patch.max(axis=(0, 1))).clip(0, 1)
    
    return image_max

def plot_white_correction(image,image_corrected,index,x_top, y_top, h,w):
    

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].add_patch(Rectangle((x_top, y_top), 
                                  w, 
                                  h, 
                                  linewidth=3,
                                  edgecolor='r', facecolor='none'));
    ax[0].imshow(cv2.cvtColor(image[index], cv2.COLOR_BGR2RGB))
    
    
    ax[0].set_title('Original image')
    
    ax[1].imshow(cv2.cvtColor(image_corrected[index], cv2.COLOR_BGR2RGB) );
    ax[1].set_title('Whitebalanced Image')


    return


def image_floattoint(float_image):
    
    float_image_rescaled = float_image * 255.0

# Step 2: Clip values to ensure they are within [0, 255]
    float_image_rescaled = np.clip(float_image_rescaled, 0, 255)
    
    # Step 3: Convert to uint8 type
    uint8_image = float_image_rescaled.astype(np.uint8)
    return uint8_image


def interpolate_one_2d(img_channel):
    
    x = np.arange(0, img_channel.shape[1])  # Width of the image
    y = np.arange(0, img_channel.shape[0])  # Height of the image
    x, y = np.meshgrid(x, y)
    
    # Flatten the arrays for interpolation
    points = np.array([x.flatten(), y.flatten()]).T
    values = img_channel.flatten()
    
    # Create a finer grid for interpolation
    x_new = np.linspace(0, img_channel.shape[1] - 1, img_channel.shape[1] * 2)  # Double the resolution
    y_new = np.linspace(0, img_channel.shape[0] - 1, img_channel.shape[0] * 2)
    x_new, y_new = np.meshgrid(x_new, y_new)
    
    # Perform the interpolation using griddata
    channel_interpolated = griddata(points, values, (x_new, y_new), method='linear')
    
    return channel_interpolated


def plot_chennel_3d(channel):
    
    x = np.arange(0, channel.shape[1])  # Width of the image
    y = np.arange(0, channel.shape[0])  # Height of the image
    x, y = np.meshgrid(x, y)
    
    # Create the 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface using the Hue channel data
    surface = ax.plot_surface(x, y, channel, cmap='hsv')

    return


def my_edge_detection(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)  # Detects edges in the X direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)  # Detects edges in the Y direction
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    
    sobel_edges = gaussian_filter(sobel_edges, sigma=30)
    sobel_edges = np.uint8(sobel_edges)
    
    contours, _ = cv2.findContours(sobel_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    
    
    output_image = np.zeros_like(image)
    
    # Step 3: Interpolate the points of each contour
    for contour in contours:
        # Extract the x and y coordinates of the contour points
        contour = contour.squeeze()  # Remove extra dimensions, contour is [n, 1, 2] -> [n, 2]
        
        # Skip if the contour has fewer than 2 points (no interpolation possible)
        if len(contour) < 10:
            continue
    
        x = contour[:, 0]  # x coordinates
        y = contour[:, 1]  # y coordinates
    
        # Linear interpolation of the contour points
        # Create interpolation functions for x and y based on their contour indices
        num_points = 100  # Number of points to interpolate along the contour
        t = np.arange(len(x))  # Parameter for original points
        t_new = np.linspace(0, len(x) - 1, num_points)  # New parameter for interpolation
    
        # Interpolation functions (could also use 'cubic' for spline interpolation)
        interp_x = interp1d(t, x, kind='linear')
        interp_y = interp1d(t, y, kind='linear')
    
        # Generate interpolated points
        x_new = interp_x(t_new)
        y_new = interp_y(t_new)
    
        # Round and convert to integers for pixel plotting
        x_new = np.round(x_new).astype(int)
        y_new = np.round(y_new).astype(int)
    
        # Step 4: Draw the interpolated points onto the output image
        for i in range(len(x_new) - 1):
            cv2.line(output_image, (x_new[i], y_new[i]), (x_new[i+1], y_new[i+1]), 255, 1)
        
    return sobel_edges,output_image

def avg_int_multipleROIs(image,w,h,c_scale):
     # Example width and height of each ROI

# Get the dimensions of the image
    image_height, image_width = image.shape
    
    # Step 2: Divide the image into ROIs and compute the average value of each
    # We will store the averages and their positions
    x_vals = []
    y_vals = []
    z_vals = []
    
    # Loop through the image and process each ROI
    for y in range(0, image_height, h):
        for x in range(0, image_width, w):
            # Define the ROI (region of interest)
            roi = image[y:y+h, x:x+w]
    
            # Compute the average pixel intensity in the ROI
            avg_intensity = np.mean(roi)
    
            # Store the x, y coordinates and the average intensity (z-axis)
            x_vals.append((x + w//2)*c_scale)  # Use the center of the ROI for plotting
            y_vals.append((y + h//2)*c_scale)
            z_vals.append(avg_intensity)
    
    # Step 3: Create the 3D plot
    x = np.asarray(x_vals)
    y = np.asarray(y_vals)
    z = np.asarray(z_vals)
    
    return x,y,z

def std_multipleROIs(image,w,h,c_scale):
     # Example width and height of each ROI

# Get the dimensions of the image
    image_height, image_width = image.shape
    
    # Step 2: Divide the image into ROIs and compute the average value of each
    # We will store the averages and their positions
    x_vals = []
    y_vals = []
    z_vals = []
    
    # Loop through the image and process each ROI
    for y in range(0, image_height, h):
        for x in range(0, image_width, w):
            # Define the ROI (region of interest)
            roi = image[y:y+h, x:x+w]
    
            # Compute the average pixel intensity in the ROI
            avg_intensity = np.std(roi)
    
            # Store the x, y coordinates and the average intensity (z-axis)
            x_vals.append((x + w//2)*c_scale)  # Use the center of the ROI for plotting
            y_vals.append((y + h//2)*c_scale)
            z_vals.append(avg_intensity)
    
    # Step 3: Create the 3D plot
    x = np.asarray(x_vals)
    y = np.asarray(y_vals)
    z = np.asarray(z_vals)
    
    return x,y,z

def avg_int_singleROI(image, x_top,y_top,w,h):
      
    roi = image[y_top:y_top+h, x_top:x_top+w]
    
            # Compute the average pixel intensity in the ROI
    avg_intensity = np.mean(roi)
    std = np.std(roi)
    
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    ax.imshow(image)
    ax.add_patch(Rectangle((x_top, y_top), 
                              w, 
                              h, 
                              linewidth=3,
                              edgecolor='r', facecolor='none'));
    ax.set_title('Original image')
       
    
    return avg_intensity,std



def lambda_normalization(arrays, new_min=340, new_max=640):
    normalized_arrays = []
    
    for arr in arrays:
        v_min = np.min(arr)  # Find min of the current array
        v_max = np.max(arr)  # Find max of the current array

        # Normalize in descending order
        normalized_arr = new_max - ((arr - v_min) / (v_max - v_min)) * (new_max - new_min)
        normalized_arrays.append(normalized_arr)
    
    return normalized_arrays

def print_first_change(array):
    # Iterate over the array
    indices = np.where(array[1:] < array[:-1])[0] + 1  # Check if current value differs from the previous one
            
    return indices

def find_deformation_field_zaxes(d_field,x_shape_img, y_shape_img,z_shape_img, dim_x, dim_y, dim_z):
    
    N = np.linspace(0,1120/.640,dim_z)
    
    x = np.linspace(0, x_shape_img, dim_x)  # X values
    y = np.linspace(0, y_shape_img, dim_y)  # Y values
    z = np.linspace(0, z_shape_img, dim_z)  # Z values
    
    # Create the meshgrid for 3D space
    X, Y, Z = np.meshgrid(x, y, z)
    
    
    a = d_field.reshape(dim_y,dim_x)
    
    
    
    # Stack the 2D matrix along the third dimension (Z axis)
    matrix_3d = np.stack([a] * dim_z, axis=-1)
    
    displacement_x = np.zeros((dim_y, dim_x, dim_z))
    displacement_y = np.zeros((dim_y, dim_x, dim_z))
    displacement_z = matrix_3d * Z/.640
    
    
    
    return X, Y, Z,displacement_x, displacement_y, displacement_z


def plot_1d_as_function_of_time(x,y,xlabel,ylabel,mrk,cmap,out_f,name):
    
    if cmap == 0:
        colors = pl.cm.viridis(np.linspace(0,1,len(y)))
    if cmap == 1:
        colors = pl.cm.magma(np.linspace(0,1,len(y)))
    if cmap == 2:
        colors = pl.cm.cool(np.linspace(0,1,len(y)))
    if cmap == 3:
        colors = pl.cm.summer(np.linspace(0,1,len(y)))
    
    
    
    plt.figure()
    ax = plt.axes()
    
    
    
    ax.plot(x,y,linestyle=':',color='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    for i in range(len(y)):
    
        ax.plot(x[i],y[i],marker=mrk,linestyle='',color=colors[i],markersize = 9,markeredgecolor='black')
        ax.set_xlabel(xlabel,fontsize=15)
        ax.set_ylabel(ylabel,fontsize=15)
        
    plt.savefig(out_f+name+'.png',dpi=300,transparent=True,bbox_inches='tight')

    
    
    return


def plot_ycut_in_time(x_list,y_list,y_value,idc,image,xlabel,ylabel,mrk,cmap,out_f,name):
    
    if cmap == 0:
        colors = pl.cm.viridis(np.linspace(0,1,len(y_value)))
    if cmap == 1:
        colors = pl.cm.magma(np.linspace(0,1,len(y_value)))
    if cmap == 2:
        colors = pl.cm.cool(np.linspace(0,1,len(y_value)))
    if cmap == 3:
        colors = pl.cm.summer(np.linspace(0,1,len(y_value)))
    
    indexes = print_first_change(x_list[0])
    
    extent = [x_list[0][indexes[idc]:indexes[idc+1]].min(), x_list[0][indexes[idc]:indexes[idc+1]].max(), y_list[0].min(), y_list[0].max()]
    
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    
    ax[0].imshow(cv2.cvtColor(image[-1], cv2.COLOR_BGR2RGB),extent=extent)
    ax[0].scatter(x_list[0][indexes[idc]:indexes[idc+1]],y_list[0][indexes[idc]:indexes[idc+1]], color='red', label='Position', s=10, marker='X')
    

    ax[0].set_title('Image')
    

    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel('y [mm]')
    
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
    
    cmap = cm.viridis
    norm = plt.Normalize(0, len(x_list) - 1) 
    
    for i in range(len(x_list)):
        ax[1].plot(x_list[i][indexes[idc]:indexes[idc+1]],y_value[i][indexes[idc]:indexes[idc+1]],marker='.',color=cmap(norm(i)),linestyle='-')
        
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Needed for older versions of Matplotlib
    cbar = plt.colorbar(sm,  label='time [s]')
    cbar.ax.set_yticklabels(np.arange(len(x_list)/14))
    
    
    ax[1].set_title('cut along y =' + str(np.round(y_list[0][indexes[idc]],2) )+ ' mm')
    plt.savefig(out_f+name+'.png',dpi=300,transparent=True,bbox_inches='tight')
    
    return


def render_frame(position, pressure):
    fig, ax = plt.subplots(figsize=(6, 4))  # Create a new figure
    canvas = FigureCanvas(fig)  # Create a canvas for rendering
    
    ax.plot(position, pressure, color='b')  # Line plot for pressure vs. position
    ax.set_title("Pressure vs Position")    # Set plot title
    ax.set_xlabel("Position")               # X-axis label
    ax.set_ylabel("Pressure")               # Y-axis label
    ax.set_xlim([min(position), max(position)])  # Fix x-axis limits
    ax.set_ylim([-2, 2])                    # Fix y-axis limits based on expected pressure range (adjust as needed)
    
    canvas.draw()  # Render the plot on the canvas
    
    # Get the RGBA buffer from the canvas and convert it to an RGB image (without alpha channel)
    width, height = canvas.get_width_height()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    
    plt.close(fig)  # Close the plot to free memory
    return image

def save_movie(x_list,y_value,idc):
    
    indexes = print_first_change(x_list[0])
    
    
    first_frame = render_frame(x_list[0][indexes[idc]:indexes[idc+1]], y_value[0][indexes[idc]:indexes[idc+1]])
    height, width, layers = first_frame.shape
    
    # Initialize the video writer (MP4 format with MJPG codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter(r'C:\Users\hp\Desktop\POST_DOC\talks_&_seminars/pressure_vs_position.mp4', fourcc, 10.0, (width, height))  # 10 FPS video
    
    # Loop through the position and pressure arrays to generate frames and write them to the video
    for i in range(len(x_list)):
        frame = render_frame(x_list[i][indexes[idc]:indexes[idc+1]], y_value[i][indexes[idc]:indexes[idc+1]])  # Render the frame from position and pressure arrays
        out.write(frame)  # Write the frame to the video
    
    # Release the video writer when done
    out.release()
    
    print("Video saved as 'pressure_vs_position.mp4'")
    return

def fit_circle(x, y, y_min=None, y_max=None):
    """Fits a circle to given x, y points using least squares, with optional y-value filtering."""
    
    # Apply filtering based on y_min and y_max
    if y_min is not None:
        mask = y >= y_min
        x, y = x[mask], y[mask]
    if y_max is not None:
        mask = y <= y_max
        x, y = x[mask], y[mask]

    # Ensure we still have enough points
    if len(x) < 3:
        raise ValueError("Not enough points remaining after filtering to fit a circle!")

    # Construct the system of equations
    A = np.column_stack((2*x, 2*y, np.ones_like(x)))
    B = x**2 + y**2

    # Solve using least squares
    a, b, c = np.linalg.lstsq(A, B, rcond=None)[0]
    center = (a, b)
    radius = np.sqrt(c + a**2 + b**2)
    
    return center, radius

def plot_circle(x, y, center, radius):
    """Plots the circle and marks the center with 'X'."""
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Plot original points
    ax.scatter(x, y, label="Given Points", color="blue", alpha=0.6)

    # Generate the fitted circle points
    theta = np.linspace(0, 2*np.pi, 300)
    circle_x = center[0] + radius * np.cos(theta)
    circle_y = center[1] + radius * np.sin(theta)
    ax.plot(circle_x, circle_y, label="Fitted Circle", color="red")

    # Mark the center with an 'X'
    ax.scatter(*center, color="black", marker='x', s=100, label="Center")

    # Labels & legend
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend()
    plt.title("Circle Fitting with Center Marked")

    plt.show()
    
def compute_tangent_normal(center, X_r, Y_r):
    """Computes the tangent and normal (orthogonal) directions at (X_r, Y_r)."""
    cx, cy = center

    # Compute normal vector (radial direction)
    normal = np.array([X_r - cx, Y_r - cy])
    normal = normal / np.linalg.norm(normal)  # Normalize

    # Compute tangent vector (perpendicular to normal)
    tangent = np.array([-normal[1], normal[0]])  # Rotate 90 degrees

    return tangent, normal

def get_rectangle_corners(X_r, Y_r, width, height, tangent, normal):
    """Computes the four corner points of the rotated rectangle."""
    dx_tangent = tangent * width / 2
    dy_normal = normal * height / 2

    corners = np.array([
        [X_r - dx_tangent[0] - dy_normal[0], Y_r - dx_tangent[1] - dy_normal[1]],
        [X_r + dx_tangent[0] - dy_normal[0], Y_r + dx_tangent[1] - dy_normal[1]],
        [X_r + dx_tangent[0] + dy_normal[0], Y_r + dx_tangent[1] + dy_normal[1]],
        [X_r - dx_tangent[0] + dy_normal[0], Y_r - dx_tangent[1] + dy_normal[1]]
    ], dtype=np.float32)

    return corners

def extract_rotated_rectangle(image, corners, width, height):
    """Extracts the rotated rectangle from the image."""
    target_corners = np.array([
        [0, 0], [width, 0], [width, height], [0, height]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, target_corners)

    # Warp the image to extract the rectangle
    extracted = cv2.warpPerspective(image, M, (width, height))

    return extracted

def compute_intensity_profiles(patch):
    """Computes the average intensity in the tangential and radial directions, ignoring zero pixels."""
    
    # Convert to float for safe division
    patch = patch.astype(float)

    # Tangential (column-wise) profile, excluding zeros
    valid_tangential = patch > 0  # Identify nonzero values
    sum_tangential = np.sum(patch * valid_tangential, axis=0)
    count_tangential = np.sum(valid_tangential, axis=0)
    tangential_profile = np.where(count_tangential > 0, sum_tangential / count_tangential, np.nan)

    # Radial (row-wise) profile, excluding zeros
    valid_radial = patch > 0
    sum_radial = np.sum(patch * valid_radial, axis=1)
    count_radial = np.sum(valid_radial, axis=1)
    radial_profile = np.where(count_radial > 0, sum_radial / count_radial, np.nan)

    return tangential_profile, radial_profile

def compute_asymmetry(profile, search_range=(-100, 100)):
    """Finds the shift that minimizes asymmetry around zero."""
    profile = np.array(profile)

    best_shift = 0
    min_asymmetry = float("inf")
    best_left, best_right = None, None

    mid = len(profile) // 2  # Initial center assumption

    for shift in range(search_range[0], search_range[1] + 1):
        shift_mid = mid + shift  # New potential symmetry center

        if shift_mid <= 0 or shift_mid >= len(profile) - 1:
            continue  # Skip invalid centers

        # Ensure even split
        left_half = profile[:shift_mid][::-1]
        right_half = profile[shift_mid:]

        # Truncate to the shortest length
        min_len = min(len(left_half), len(right_half))
        left_half = left_half[:min_len]
        right_half = right_half[:min_len]

        # Compute asymmetry
        asymmetry = np.sum(np.abs(left_half - right_half)) / np.sum(left_half + right_half + 1e-8)

        # Update best shift
        if asymmetry < min_asymmetry:
            min_asymmetry = asymmetry
            best_shift = shift
            best_left, best_right = left_half, right_half

    return best_shift, min_asymmetry, best_left, best_right


def plot_rectangles(image, corners_tangential, corners_radial):
    """Plots the image, both rectangles, intensity profiles, and asymmetry visualization."""
    plt.figure(figsize=(12, 7))

    # Image with both rectangles
    
    plt.imshow(image, cmap='gray')
    rect_tang = np.vstack([corners_tangential, corners_tangential[0]])  
    rect_rad = np.vstack([corners_radial, corners_radial[0]])  
    plt.plot(rect_tang[:, 0], rect_tang[:, 1], 'r-', linewidth=2, label="Tangential Rect")
    plt.plot(rect_rad[:, 0], rect_rad[:, 1], 'g-', linewidth=2, label="Radial Rect")
    
    plt.legend()
    plt.title("Image with Both Rectangles")

    plt.tight_layout()
    plt.show()
    return
    
def plot_asymmetry(profile,optimal_shift_t,min_asymmetry_t,left_half,right_half,col):
    plt.figure(figsize=(8, 8))

    x_for_plot = np.linspace(-len(profile)//2,len(profile)//2,len(profile))
    
    plt.subplot(2, 2, 1)
    plt.plot(x_for_plot,profile, label="Intensity Profile"'r-',color=col)
    plt.xlabel("Position")
    plt.ylabel("Intensity")
    plt.title(f"Optimized Symmetry (Shift: {optimal_shift_t}, Asym: {min_asymmetry_t:.4f})")
    plt.legend()
    
    
    
    plt.subplot(2, 2, 2)
    plt.plot(left_half, '--',color=col, label="Mirrored Left Half")
    plt.plot(right_half, '-',color=col, label="Original Right Half")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.title("Best Symmetry Comparison")
    plt.legend()
    
    return

def mak_for_pressure_calibration(tolerance,img_0,img):
    
    # Create a mask where the absolute difference is greater than the tolerance.
    mask = (-img_0 + img > tolerance).astype(np.uint8)
    
    
    mask_sym = np.flip(mask, axis=(0, 1))
    
    # Only keep pixels that are '1' in both the original mask and its symmetric copy.
    mask_center_symmetric = (mask & mask_sym).astype(np.uint8)
    
    # 2. Enforce Continuity
    # -----------------------------------
    # (a) Use morphological operations to remove small isolated points and fill holes.
    kernel = np.ones((3, 3), np.uint8)
    # First, a closing operation fills small holes in the mask.
    mask_closed = cv2.morphologyEx(mask_center_symmetric, cv2.MORPH_CLOSE, kernel)
    # Then, an opening operation removes small isolated regions/noise.
    mask_refined = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    
    # (b) Alternatively, you can keep only the largest continuous region (if that is expected):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_refined, connectivity=8)
    if num_labels > 1:
        # Ignore the background label (label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask = (labels == largest_label).astype(np.uint8)
    else:
        final_mask = mask_refined
    
    # 3. Visualize the Final Refined Mask
    
    
    mask_for_flood = (final_mask * 255).astype(np.uint8)
    h, w = mask_for_flood.shape
    
    # Create a temporary mask for floodFill (needs to be 2 pixels larger than the image)
    floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Flood fill from a border pixel. Here, we choose the top-left corner (0,0)
    cv2.floodFill(mask_for_flood, floodfill_mask, (0, 0), 255)
    
    # Invert the flood-filled image. This will make the background (that was reached) 0,
    # and the holes (which were not reached) 255.
    mask_inverted = cv2.bitwise_not(mask_for_flood)
    
    # The holes are now in mask_inverted. Combine them with your original final_mask:
    # Any pixel in mask_inverted that is 255 corresponds to a hole, so set it to 1.
    filled_mask = final_mask.copy()
    filled_mask[mask_inverted == 255] = 1
    

    
    return final_mask


def compute_circle_parameters(center, point):
    """
    Compute the circle parameters from the center and a point on the circle.
    
    Parameters:
      center : Tuple (X0, Y0)
      point  : Tuple (X, Y)
      
    Returns:
      r    : Radius (distance between center and point)
      phi0 : Central angle in radians (angle of the vector from center to point)
    """
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    r = np.sqrt(dx**2 + dy**2)
    phi0 = np.arctan2(dy, dx)
    return r, phi0


def extract_arc_patch(image, center, point, l, h, n_r=None, n_phi=None):
    """
    Extracts an image patch corresponding to a circular arc.
    
    Parameters:
      image   : Input grayscale image (2D numpy array)
      center  : Tuple (X0, Y0) giving the center of the circle.
      point   : Tuple (X, Y) lying on the circle (defines both radius and phi0).
      l       : Arc length (in pixels) along the circle.
                (The angular extent is computed as θ = l/r.)
      h       : Thickness of the arc (in pixels, in the radial direction).
      n_r     : Number of radial samples. If None, uses ceil(h).
      n_phi   : Number of angular (tangential) samples. If None, uses ceil(l).
      
    Returns:
      arc_patch  : Extracted patch (2D array) of shape (n_r, n_phi) where rows span the radial direction
                   (from r - h/2 to r + h/2) and columns span the angular direction.
      R_grid, phi_grid : The 2D grids of sampled polar coordinates.
      r_values, phi_values : The 1D arrays of sampled radii and angles.
    """
    # Compute radius and central angle from center and point.
    r, phi0 = compute_circle_parameters(center, point)
    
    if n_r is None:
        n_r = int(np.ceil(h))
    if n_phi is None:
        n_phi = int(np.ceil(l))
    
    theta = l / r  # angular extent in radians

    # Define the sampled radial and angular coordinates:
    r_values = np.linspace(r - h/2, r + h/2, n_r)
    phi_values = np.linspace(phi0 - theta/2, phi0 + theta/2, n_phi)
    
    # Create a 2D grid of polar coordinates (rows: radial, columns: angular):
    phi_grid, R_grid = np.meshgrid(phi_values, r_values)
    
    # Convert polar coordinates (with respect to center) to Cartesian image coordinates.
    X0, Y0 = center
    x_map = X0 + R_grid * np.cos(phi_grid)
    y_map = Y0 + R_grid * np.sin(phi_grid)
    
    # cv2.remap requires float32 maps.
    map_x = x_map.astype(np.float32)
    map_y = y_map.astype(np.float32)
    
    # Use remap to sample the image; pixels falling outside the image become 0.
    arc_patch = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return arc_patch, R_grid, phi_grid, r_values, phi_values

def compute_intensity_profiles_arc(patch):
    """
    Computes average intensity profiles along the tangential (angular) and radial directions
    from the arc patch. Zero pixels (which might be due to out-of-bound sampling) are ignored.
    
    Parameters:
      patch : 2D numpy array (arc patch)
      
    Returns:
      tangential_profile : 1D array averaged over the radial direction (for each angular column)
      radial_profile     : 1D array averaged over the angular direction (for each radial row)
    """
    patch = patch.astype(float)
    
    # Compute tangential profile: average each column, ignoring zeros.
    tangential_profile = []
    for j in range(patch.shape[1]):
        col = patch[:, j]
        valid = col != 0
        if np.any(valid):
            tangential_profile.append(np.mean(col[valid]))
        else:
            tangential_profile.append(np.nan)
    tangential_profile = np.array(tangential_profile)
    
    # Compute radial profile: average each row, ignoring zeros.
    radial_profile = []
    for i in range(patch.shape[0]):
        row = patch[i, :]
        valid = row != 0
        if np.any(valid):
            radial_profile.append(np.mean(row[valid]))
        else:
            radial_profile.append(np.nan)
    radial_profile = np.array(radial_profile)
    
    return tangential_profile, radial_profile

def plot_arc_on_image(image, center, point, l, h):
    """
    Overlays the boundaries of the arc on the original image for visualization.
    
    The boundaries are computed from the center and the user-provided point.
    """
    X0, Y0 = center
    r, phi0 = compute_circle_parameters(center, point)
    theta = l / r  # angular extent
    
    # Compute inner and outer boundaries of the arc.
    phi_boundary = np.linspace(phi0 - theta/2, phi0 + theta/2, 200)
    inner_r = r - h/2
    outer_r = r + h/2
    
    inner_x = X0 + inner_r * np.cos(phi_boundary)
    inner_y = Y0 + inner_r * np.sin(phi_boundary)
    outer_x = X0 + outer_r * np.cos(phi_boundary)
    outer_y = Y0 + outer_r * np.sin(phi_boundary)
    
    # Compute the side (radial) boundaries at the angular limits.
    radial_r = np.linspace(r - h/2, r + h/2, 100)
    side1_x = X0 + radial_r * np.cos(phi0 - theta/2)
    side1_y = Y0 + radial_r * np.sin(phi0 - theta/2)
    side2_x = X0 + radial_r * np.cos(phi0 + theta/2)
    side2_y = Y0 + radial_r * np.sin(phi0 + theta/2)
    
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.plot(inner_x, inner_y, 'r-', label='Inner boundary')
    plt.plot(outer_x, outer_y, 'r-', label='Outer boundary')
    plt.plot(side1_x, side1_y, 'r-', label='Side boundaries')
    plt.plot(side2_x, side2_y, 'r-')
    plt.scatter([X0], [Y0], color='yellow', marker='x', s=100, label='Center')
    plt.title("Arc Boundaries Overlaid on Image")
    plt.legend()
    plt.show()

def extract_rect_patch(image, center, point, l, h, n_r=None, n_phi=None):
    """
    Extracts an image patch corresponding to a rectangle of dimensions h and l.
    
    The patch is defined as follows:
      - It is centered at the given 'point'.
      - Its vertical axis (of length h) is aligned with the radial direction from 'center' to 'point'.
      - Its horizontal axis (of length l) is perpendicular to the radial direction (i.e. tangential).
      
    Parameters:
      image   : Input grayscale image (2D numpy array)
      center  : Tuple (X0, Y0) giving the center.
      point   : Tuple (X, Y) at which the patch is centered (also defines orientation).
      l       : Length of the rectangle in the tangential direction (pixels).
      h       : Height of the rectangle in the radial direction (pixels).
      n_r     : Number of samples along the radial direction. If None, uses ceil(h).
      n_phi   : Number of samples along the tangential direction. If None, uses ceil(l).
      
    Returns:
      patch      : Extracted patch (2D array) of shape (n_r, n_phi) where rows span the radial offsets
                   (from -h/2 to h/2) and columns span the tangential offsets (from -l/2 to l/2).
      R_grid, phi_grid : 2D grids of the sampling coordinates (R_grid: radial offset, phi_grid: tangential offset).
      r_values, phi_values : 1D arrays of the sampled radial and tangential coordinates.
    """
    # Compute orientation from center to point (we only need the angle φ₀)
    _, phi0 = compute_circle_parameters(center, point)
    
    if n_r is None:
        n_r = int(np.ceil(h))
    if n_phi is None:
        n_phi = int(np.ceil(l))
        
    # Define 1D coordinate arrays:
    # 'r_values' corresponds to radial offsets (vertical direction)
    # 'phi_values' corresponds to tangential offsets (horizontal direction)
    r_values = np.linspace(-h/2, h/2, n_r)
    phi_values = np.linspace(-l/2, l/2, n_phi)
    
    # Create a 2D grid of sampling coordinates.
    # (Rows correspond to radial offset; columns correspond to tangential offset.)
    phi_grid, R_grid = np.meshgrid(phi_values, r_values)
    
    # Define unit vectors for the radial and tangential directions.
    radial_vector = np.array([np.cos(phi0), np.sin(phi0)])
    tangent_vector = np.array([-np.sin(phi0), np.cos(phi0)])
    
    # Map each grid point (phi, R) in the patch coordinate system to image coordinates.
    # Here the patch is centered at 'point'.
    x_map = point[0] + phi_grid * tangent_vector[0] + R_grid * radial_vector[0]
    y_map = point[1] + phi_grid * tangent_vector[1] + R_grid * radial_vector[1]
    
    # cv2.remap requires mapping arrays of type float32.
    map_x = x_map.astype(np.float32)
    map_y = y_map.astype(np.float32)
    
    # Extract the patch from the image; pixels falling outside the image become 0.
    patch = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return patch, R_grid, phi_grid, r_values, phi_values

def plot_rect_on_image(image, center, point, l, h):
    """
    Overlays the boundaries of a rectangular patch on the original image for visualization.
    
    The rectangle is defined as:
      - Centered at the provided 'point'.
      - With width l (tangential direction) and height h (radial direction).
      - Its vertical (radial) axis is aligned with the vector from 'center' to 'point'.
      - Its horizontal (tangential) axis is perpendicular to that vector.
    
    Parameters:
      image  : Input image (grayscale or color).
      center : Tuple (X0, Y0) representing the circle (or reference) center.
      point  : Tuple (X, Y) at which the patch is centered.
      l      : Length of the rectangle in the tangential direction (pixels).
      h      : Height of the rectangle in the radial direction (pixels).
    """
    X0, Y0 = center
    # Compute the angle from center to point.
    _, phi0 = compute_circle_parameters(center, point)
    
    # Define unit vectors:
    # Radial direction (from center to point)
    radial_vector = np.array([np.cos(phi0), np.sin(phi0)])
    # Tangential direction (perpendicular to the radial vector)
    tangent_vector = np.array([-np.sin(phi0), np.cos(phi0)])
    
    # Half-dimensions:
    half_l = l / 2.0
    half_h = h / 2.0
    
    # Compute the four corners of the rectangle.
    # Order: bottom-left, bottom-right, top-right, top-left.
    p = np.array(point)
    corner1 = p - half_l * tangent_vector - half_h * radial_vector
    corner2 = p + half_l * tangent_vector - half_h * radial_vector
    corner3 = p + half_l * tangent_vector + half_h * radial_vector
    corner4 = p - half_l * tangent_vector + half_h * radial_vector
    
    # Stack the corners and repeat the first to close the loop.
    corners = np.array([corner1, corner2, corner3, corner4, corner1])
    
    plt.figure(figsize=(6,6))
    # Convert image to RGB if it is color.
    if len(image.shape) == 3 and image.shape[2] == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    
    # Plot the rectangle boundaries.
    plt.plot(corners[:, 0], corners[:, 1], 'r-', linewidth=2, label='Rectangle Boundary')
    # Mark the patch center.
    plt.scatter([point[0]], [point[1]], color='yellow', marker='x', s=100, label='Patch Center')
    # Optionally, mark the given circle center.
    plt.scatter([X0], [Y0], color='cyan', marker='o', s=100, label='Circle Center')
    plt.title("Rectangle Boundaries Overlaid on Image")
    plt.legend()
    plt.show()
    
    
def calculate_profiles_indentation(smoothed_hue,center,half_length):
    
    # Assume 'image' is your grayscale image as a 2D NumPy array.
    # Define the center point (row, col) for the lines.

    
    # Define the half-length of the line (in pixels). Each line extends half_length pixels in both directions.

    # Define the number of points per profile (fixed for all lines).
    profile_length = 2 * half_length + 1
    
    # Define the number of lines (different orientations) to sample.
    num_angles = 100
    # Use angles between 0 and pi (0° to 180°) since a line and its opposite are equivalent.
    angles = np.linspace(0, np.pi, num=num_angles, endpoint=False)
    
    avg_profiles = []
    distances = []
    for i in range(len(smoothed_hue)):
        
        image = smoothed_hue[i]
    
    # List to store each profile (each will have 'profile_length' points).
        profiles = []
        
        # Loop over each angle to extract the profile.
        for theta in angles:
            # Compute the directional vector components.
            dr = np.sin(theta)  # change in row (vertical)
            dc = np.cos(theta)  # change in column (horizontal)
            
            # Compute the start and end coordinates so that the line is centered at 'center'.
            start = (center[0] - half_length * dr, center[1] - half_length * dc)
            end   = (center[0] + half_length * dr, center[1] + half_length * dc)
            
            # Generate coordinates along the line with a fixed number of points.
            rows = np.linspace(start[0], end[0], profile_length)
            cols = np.linspace(start[1], end[1], profile_length)
            
            # Interpolate the intensity values along the line.
            profile = map_coordinates(image, [rows, cols], order=1, mode='reflect')
            profiles.append(profile)
        
        # Convert the list of profiles to a 2D NumPy array (shape: num_angles x profile_length).
        profiles = np.array(profiles)
        
        # Compute the average profile across all angles.
        avg_profile = profiles.mean(axis=0)
        avg_profiles.append(avg_profile)
        # Create a distance axis (with 0 at the center) for plotting.
        distance = np.linspace(-half_length, half_length, profile_length)
        distances.append(distance)
    
    
    return distances,avg_profiles


def get_blob_info(image, min_sigma, max_sigma, num_iso, 
                  sigma_smooth=2, blob_threshold=0.1, threshold_factor=0.5):
    """
    Detects a blob (hill) in the image, refines its center,
    and computes iso‐heights (gray levels spanning the blob’s intensity range).

    Parameters:
      image           : 2D input image.
      min_sigma       : Minimum sigma for blob detection.
      max_sigma       : Maximum sigma for blob detection.
      num_iso         : Number of iso‐height levels.
      sigma_smooth    : Sigma for Gaussian smoothing.
      blob_threshold  : Threshold for blob_log.
      threshold_factor: Fraction of the peak intensity to define the blob region.

    Returns:
      iso_heights   : 1D numpy array of iso‐values.
      peak_value    : Refined peak intensity.
      peak_position : Refined peak position (row, col) as a tuple.
      smoothed      : The smoothed image.
    """
    # Smooth the image.
    smoothed = gaussian_filter(image, sigma=sigma_smooth)

    # Detect blobs using Laplacian of Gaussian.
    blobs = blob_log(smoothed, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=10, threshold=blob_threshold)
    blobs = blobs[blobs[:, 2] > min_sigma]  # filter out very small blobs
    if blobs.shape[0] == 0:
        raise ValueError("No blobs detected. Adjust blob_log parameters.")

    # Choose the blob with the largest sigma.
    blob = blobs[np.argmax(blobs[:, 2])]
    initial_center = (int(blob[0]), int(blob[1]))
    initial_peak = smoothed[initial_center]

    # Define blob region using a threshold (fraction of initial peak).
    region_threshold = initial_peak * threshold_factor
    iso_mask = smoothed >= region_threshold
    labeled_mask, _ = label(iso_mask)
    component_label = labeled_mask[initial_center]
    if component_label == 0:
        raise ValueError("Initial center not inside any blob region. Adjust threshold_factor.")
    blob_region = (labeled_mask == component_label)

    # Refine the center: choose the pixel of maximum intensity within the blob region.
    peak_position = np.unravel_index(np.argmax(smoothed * blob_region), smoothed.shape)
    peak_value = smoothed[peak_position]

    # Determine minimum intensity in the blob region.
    min_blob_intensity = np.min(smoothed[blob_region])
    # Compute iso‐heights from min intensity up to the peak.
    iso_heights = np.linspace(min_blob_intensity, peak_value, num=num_iso)

    return iso_heights, peak_value, peak_position, smoothed

#########################################################################
# 2. Function: Compute Region Value Difference Using the Reference Line
#########################################################################
def compute_region_value_difference(iso_value, smoothed, peak_position, ref_point, center_mode='peak'):
    """
    For a given iso-value, partitions the iso-region (the connected component with
    intensity >= iso_value that contains the blob's peak) using the line joining a center
    and the reference. The center can be chosen as the blob's peak ("peak") or the geometric
    center (mean) of the iso-contour ("geo").

    Pixels in the region are partitioned by computing, for each pixel p (with (x,y) coordinates,
    where x = column, y = row), the signed value of:
         f(p) = a * p_x + b * p_y + c,
    where the line joining center and reference is given by:
         a = (y_ref - y_center),
         b = (x_center - x_ref),
         c = (x_ref*y_center - x_center*y_ref).

    With our convention here, pixels with f(p) < 0 are “front” (to the right of the line when moving from center to reference)
    and those with f(p) >= 0 are “back.”

    The normalized total difference is computed as:
         (sum(front pixel values) - sum(back pixel values)) / (N_front + N_back).

    Parameters:
      iso_value    : The gray value defining the iso-region.
      smoothed     : The smoothed image.
      peak_position: Blob peak as (row, col) (tuple).
      ref_point    : Reference point in (x, y) coordinates.
      center_mode  : "peak" (default) to use the blob's peak or "geo" to use the geometric center of the iso-contour.

    Returns:
      value_diff : The computed normalized total difference.
      front_mask : Boolean mask for the front region.
      back_mask  : Boolean mask for the back region.
      iso_contour: The chosen iso-contour (as a numpy array of (x, y) points).
    """
    # Threshold the image to get the iso-region.
    iso_mask = smoothed >= iso_value
    labeled_mask, _ = label(iso_mask)
    comp_label = labeled_mask[peak_position]
    if comp_label == 0:
        raise ValueError("Peak not in an iso-region at iso_value={:.2f}".format(iso_value))
    region_mask = (labeled_mask == comp_label)

    # Get indices (rows, cols) of pixels in the region.
    rows, cols = np.where(region_mask)
    points = np.column_stack((cols, rows))  # (x,y) coordinates

    # Compute the center to use.
    peak_xy = np.array([peak_position[1], peak_position[0]], dtype=float)
    if center_mode == 'geo':
        # Compute iso-contour that encloses the peak.
        contours = find_contours(smoothed, level=iso_value)
        iso_contour = None
        for cnt in contours:
            cnt_xy = cnt[:, [1, 0]]  # convert (row, col) -> (x,y)
            if Path(cnt_xy).contains_point(peak_xy):
                iso_contour = cnt_xy
                break
        if iso_contour is None:
            raise ValueError("No iso-contour found enclosing the peak at iso_value={:.2f}".format(iso_value))
        center = np.mean(iso_contour, axis=0)
    else:
        center = peak_xy
        # Also compute iso_contour in this mode.
        contours = find_contours(smoothed, level=iso_value)
        iso_contour = None
        for cnt in contours:
            cnt_xy = cnt[:, [1, 0]]
            if Path(cnt_xy).contains_point(peak_xy):
                iso_contour = cnt_xy
                break
        if iso_contour is None:
            raise ValueError("No iso-contour found enclosing the peak at iso_value={:.2f}".format(iso_value))

    # Compute line parameters from the chosen center to the reference.
    ref_arr = np.array(ref_point, dtype=float)
    # The line passing through center and reference has coefficients:
    # a = (y_ref - y_center), b = (x_center - x_ref), c = (x_ref*y_center - x_center*y_ref)
    a = ref_arr[1] - center[1]
    b = center[0] - ref_arr[0]
    c = ref_arr[0]*center[1] - center[0]*ref_arr[1]
    # For each point p in the region, compute f(p) = a*p_x + b*p_y + c.
    f_vals = a*points[:,0] + b*points[:,1] + c

    # Determine front/back based on a test point: move a small step from the center toward the reference.
    eps = 1e-3
    d = ref_arr - center
    d_norm = d / np.linalg.norm(d)
    test_pt = center + eps*d_norm
    f_test = a*test_pt[0] + b*test_pt[1] + c
    # If f_test < 0 then "front" is defined as f(p) < 0.
    if f_test < 0:
        front_inds = f_vals < 0
        back_inds = f_vals >= 0
    else:
        front_inds = f_vals >= 0
        back_inds = f_vals < 0

    if np.sum(front_inds) == 0 or np.sum(back_inds) == 0:
        raise ValueError("Insufficient pixels on one side of the dividing line.")

    # Compute total pixel values in front and back.
    front_pixels = smoothed[rows[front_inds], cols[front_inds]]
    back_pixels  = smoothed[rows[back_inds], cols[back_inds]]
    front_total = np.sum(front_pixels)
    back_total = np.sum(back_pixels)
    
    value_diff = np.abs(front_total/len(front_pixels) - back_total/len(back_pixels)) 

    # Create full-image boolean masks for front and back.
    front_mask = np.zeros(smoothed.shape, dtype=bool)
    back_mask  = np.zeros(smoothed.shape, dtype=bool)
    front_mask[rows[front_inds], cols[front_inds]] = True
    back_mask[rows[back_inds], cols[back_inds]] = True

    return value_diff, front_mask, back_mask, iso_contour



def compute_global_asymmetry(smoothed, peak_position, iso_value):
    """
    Computes a global asymmetry measure for the blob region defined by the iso_value.
    The asymmetry is calculated by comparing the blob to its 180° rotated (mirrored) version.
    
    Parameters:
      smoothed     : The smoothed image.
      peak_position: Blob peak as (row, col) (tuple).
      iso_value    : The iso-value to threshold the image and extract the blob.
      
    Returns:
      asymmetry   : The global asymmetry value (lower means more symmetric).
      subimage    : The extracted subimage of the blob.
      I_rot       : The 180° rotated version of the subimage.
      submask     : Boolean mask indicating the blob region in the subimage.
    """
    # Threshold the image at the iso_value.
    iso_mask = smoothed >= iso_value
    labeled_mask, _ = label(iso_mask)
    comp_label = labeled_mask[peak_position]
    if comp_label == 0:
        raise ValueError("Peak not in an iso-region at iso_value={:.2f}".format(iso_value))
    blob_mask = (labeled_mask == comp_label)
    
    # Extract region properties using regionprops.
    props = regionprops(blob_mask.astype(int), intensity_image=smoothed)
    if len(props) == 0:
        raise ValueError("No region properties found.")
    prop = props[0]
    min_row, min_col, max_row, max_col = prop.bbox
    # Extract the subimage and corresponding mask.
    subimage = smoothed[min_row:max_row, min_col:max_col]
    submask = blob_mask[min_row:max_row, min_col:max_col]
    # Rotate the subimage by 180° (flip vertically and horizontally).
    I_rot = np.flipud(np.fliplr(subimage))
    
    # Compute asymmetry over the blob region (only consider pixels in submask).
    I_region = subimage[submask]
    I_rot_region = I_rot[submask]
    asymmetry = np.sum(np.abs(I_region - I_rot_region)) / np.sum(I_region)
    return asymmetry, subimage, I_rot, submask






def fit_ellipse_and_oblateness(contour, normalization='difference'):
    """
    Fit an ellipse to the iso-contour points and compute the oblateness.
    
    The oblateness is defined as:
       (major_axis - minor_axis) / major_axis
    
    Parameters:
      iso_contour : ndarray of shape (N, 2)
         Contour points as (row, col) coordinates.
      normalization : str, optional
         If 'ratio', returns minor/major. If 'difference', returns (major - minor)/major.
         Default is 'difference'.
    
    Returns:
      ellipse_params : tuple or None
         (xc, yc, a, b, theta) where (xc, yc) is the center in (col, row) order,
         a and b are the semi-axis lengths with a >= b, and theta is the rotation (in radians).
         Returns None if the ellipse cannot be estimated.
      oblate_value : float or None
         The computed oblateness value.
         If normalization == 'ratio', returns b/a.
         If normalization == 'difference', returns (a - b)/a.
         Returns None if estimation failed.
    """
    # Convert contour points from (row, col) to (x, y) where x=col, y=row
    points = contour[:, [1, 0]]
    
    ellipse_model = EllipseModel()
    success = ellipse_model.estimate(points)
    if not success:
        return None, None
    xc, yc, a, b, theta = ellipse_model.params
    # Ensure a is the major semi-axis
    if b > a:
        a, b = b, a
    if normalization == 'ratio':
        oblate_value = b / a
    elif normalization == 'difference':
        oblate_value = (a - b) / a
    else:
        oblate_value = None
    return (xc, yc, a, b, theta), oblate_value

def binning(x,y,num_bins):

    bins = np.linspace(x.min(), x.max(), num_bins+1)

    # Get the bin indices for each theta value
    bin_indices = np.digitize(x, bins)
    
    # Initialize an array to hold the average intensity for each bin
    avg_intensity = np.zeros(num_bins)
    std_intensity = np.zeros(num_bins)
    # Compute the bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # Loop over each bin to compute the mean intensity
    for i in range(1, num_bins+1):
        # Select the intensities that fall into the current bin
        mask = bin_indices == i
        if np.any(mask):
            avg_intensity[i-1] = y[mask].mean()
            std_intensity[i-1] = y[mask].std()
        else:
            avg_intensity[i-1] = np.nan
            std_intensity[i-1] = np.nan
    return bin_centers,avg_intensity,std_intensity


def mask_circle(image: np.ndarray, x0: int, y0: int, r: int) -> np.ndarray:
    """
    Zero out a circular region of radius r centered at (x0, y0) in the input image.
    """
    result = image.copy()
    mask = np.zeros(result.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(x0), int(y0)), int(r), 255, thickness=-1)
    result[mask == 255] = 0
    return result

def compute_gradients(image: np.ndarray, ksize: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute image gradients using the Sobel operator.

    Parameters:
        image (np.ndarray): Input image as a single-channel or color image.
        ksize (int): Kernel size for the Sobel operator (1, 3, 5, or 7).

    Returns:
        grad_x (np.ndarray): Gradient in the x-direction.
        grad_y (np.ndarray): Gradient in the y-direction.
        magnitude (np.ndarray): Gradient magnitude.
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return grad_x, grad_y, magnitude


def compute_radial_gradient(image: np.ndarray, x0: float, y0: float, ksize: int = 3) -> np.ndarray:
    """
    Compute the radial gradient of an image starting from a center point (x0, y0).
    The radial gradient at each pixel is the directional derivative of the image
    intensity along the line from the center to that pixel.

    Parameters:
        image (np.ndarray): Input image as single-channel or color.
        x0 (float): X-coordinate of the center point.
        y0 (float): Y-coordinate of the center point.
        ksize (int): Kernel size for the Sobel operator.

    Returns:
        radial_grad (np.ndarray): Radial gradient image (float64).
    """
    # Compute Cartesian gradients
    grad_x, grad_y, _ = compute_gradients(image, ksize=ksize)

    # Build coordinate grids
    h, w = grad_x.shape
    xs = np.arange(w)
    ys = np.arange(h)
    xv, yv = np.meshgrid(xs, ys)

    # Compute radial vectors (from center to each pixel)
    dx = xv.astype(np.float64) - x0
    dy = yv.astype(np.float64) - y0
    r = np.sqrt(dx**2 + dy**2)

    # Avoid division by zero at center
    r_safe = np.where(r == 0, 1.0, r)

    # Unit radial vectors
    ux = dx / r_safe
    uy = dy / r_safe

    # Dot product of gradient with unit radial direction
    radial_grad = grad_x * ux + grad_y * uy
    return radial_grad

def tamplate_match_box(img,template):
    
    
    res = cv2.matchTemplate(img, template, 3)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    h, w = template.shape[:2] 
    
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)  # Moved this line up before using it
    
    bbox  = (top_left[0],top_left[0]+w,top_left[1],top_left[1]+h)
    return bbox


def region_growing(image, seed_threshold=200, connectivity=1):
    """
    Segment high-intensity regions using region growing.

    Parameters:
        image : 2D numpy array
            Grayscale image.
        seed_threshold : int
            Pixels above this value are used as seeds.
        connectivity : int
            1 for 4-connectivity, 2 for 8-connectivity.

    Returns:
        labeled_regions : 2D numpy array
            Each connected high-intensity region is assigned a unique integer.
        num_regions : int
            Number of detected regions.
    """
    # Step 1: Create a binary mask of high-intensity pixels
    seeds = image > seed_threshold

    # Step 2: Define neighborhood connectivity
    structure = generate_binary_structure(2, connectivity)

    # Step 3: Label connected components (region growing)
    labeled_regions, num_regions = label(seeds, structure=structure)
    print('ciao')
    
    region_sizes = [(np.sum(labeled_regions == i), i) for i in range(1, num_regions+1)]
    largest_region_id = max(region_sizes)[1]
    largest_region_mask = (labeled_regions == largest_region_id)
    
    edges = feature.canny(largest_region_mask.astype(float))

    return edges

def fit_fourier_for_contact_area(edges, order_expasion):
    
    N = order_expasion
    ys, xs = np.nonzero(edges)  # rows (y), cols (x)
    if len(xs) == 0:
        raise RuntimeError("No edge pixels found. Try changing Canny thresholds.")
    
    # Compute centroid as mean of edge points
    cx = xs.mean()
    cy = ys.mean()
    
    # Convert to polar coordinates relative to centroid
    theta = np.arctan2(ys - cy, xs - cx)  # range [-pi, pi]
    r = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    
    order = np.argsort(theta)
    theta_sorted = theta[order]
    r_sorted = r[order]
    
    # Build design matrix using only cosine terms
    # r(theta) = a0 + sum_{n=1..N} a_n * cos(n*theta)
    A = np.ones((theta_sorted.size, N+1))
    for n in range(1, N+1):
        A[:, n] = np.cos(n * theta_sorted)
    
    # Least squares solve for coefficients
    coeffs, *_ = lstsq(A, r_sorted, rcond=None)  # coeffs[0]=a0, coeffs[1]=a1, ...
    
    # Reconstruct r on fine theta grid for smooth curve
    theta_fit = np.linspace(-np.pi, np.pi, 1000)
      # number of Fourier harmonics
    
    # Design matrix
    A = np.column_stack([np.ones_like(theta)] +
                        [f(theta) for n in range(1, N+1) for f in (lambda t, n=n: cos(n*t), lambda t, n=n: sin(n*t))])
    
    # Solve least squares
    coeffs, _, _, _ = lstsq(A, r, rcond=None)
    
    theta_fit = np.linspace(-np.pi, np.pi, 720)
    A_fit = np.column_stack([np.ones_like(theta_fit)] +
                            [f(theta_fit) for n in range(1, N+1) for f in (lambda t, n=n: cos(n*t), lambda t, n=n: sin(n*t))])
    r_fit = A_fit @ coeffs
    
    x_fit = cx + r_fit * np.cos(theta_fit)
    y_fit = cy + r_fit * np.sin(theta_fit)
    
    
    a0 = coeffs[0]
    an = coeffs[1::2]  # a1, a2, ...
    bn = coeffs[2::2]  # b1, b2, ...
    
    # ===== Compute area =====
    area = np.pi * (a0**2 + 0.5 * np.sum(an**2 + bn**2))
    #print("Enclosed area:", area)
    
    # ===== Compute oblateness / deviation from circle =====
    oblateness = np.sqrt(np.sum(an**2 + bn**2)) / a0
    #print("Oblateness:", oblateness)
    
    
    return coeffs,x_fit,y_fit,area,oblateness








