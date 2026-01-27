# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 16:12:25 2026

@author: mateo
"""

import numpy as np
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def parse_frd_nodes(file_path):
    """
    Extract nodes from a CalculiX/PrePoMax .frd file.
    Returns Nx4 array: nodeID, x, y, z
    """
    nodes = []
    float_pattern = re.compile(r'[-+]?\d*\.\d+E[-+]?\d+')
    int_pattern = re.compile(r'\d+')
    
    with open(file_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('-1'):
                # Remove "-1" prefix
                raw = stripped[2:].strip()
                # Node ID
                m_id = int_pattern.match(raw)
                if not m_id:
                    continue
                node_id = int(m_id.group())
                rest = raw[m_id.end():]
                # Coordinates: three floats
                coords = float_pattern.findall(rest)
                if len(coords) < 3:
                    continue
                x, y, z = map(float, coords[:3])
                nodes.append([node_id, x, y, z])
    return np.array(nodes)

def parse_frd_displacements(file_path):
    """
    Extract node displacements from a CalculiX/PrePoMax .frd file.
    Returns Nx4 array: nodeID, Ux, Uy, Uz
    """
    disp = []
    float_pattern = re.compile(r'[-+]?\d*\.\d+E[-+]?\d+')
    int_pattern = re.compile(r'\d+')
    reading_disp = False
    
    with open(file_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            # Start reading displacements section
            if stripped.startswith('-4') and 'DISP' in stripped:
                reading_disp = True
                continue
            if stripped.startswith('-4') and reading_disp:
                # End of DISP section
                break
            if reading_disp and stripped.startswith('-1'):
                # Node displacement line
                raw = stripped[2:].strip()
                m_id = int_pattern.match(raw)
                if not m_id:
                    continue
                node_id = int(m_id.group())
                rest = raw[m_id.end():]
                values = float_pattern.findall(rest)
                if len(values) < 3:
                    continue
                Ux, Uy, Uz = map(float, values[:3])
                disp.append([node_id, Ux, Uy, Uz])
    
    return np.array(disp)

def parse_inp_nodes_elements(file_inp):
    nodes = []
    elements = []
    reading_nodes = False
    reading_elements = False
    
    with open(file_inp,'r') as f:
        for line in f:
            line_strip = line.strip()
            if not line_strip or line_strip.startswith('**'):
                continue  # skip comments/empty
            if line_strip.upper().startswith('*NODE'):
                reading_nodes = True
                reading_elements = False
                continue
            elif line_strip.upper().startswith('*ELEMENT'):
                reading_nodes = False
                reading_elements = True
                continue
            elif line_strip.startswith('*'):
                reading_nodes = False
                reading_elements = False
                continue
            
            if reading_nodes:
                # Only parse lines starting with a digit
                if line_strip[0].isdigit():
                    parts = line_strip.split(',')
                    node_id = int(parts[0])
                    x, y, z = map(float, parts[1:4])
                    nodes.append([node_id, x, y, z])
            elif reading_elements:
                # Only parse lines starting with a digit
                if line_strip[0].isdigit():
                    parts = line_strip.split(',')
                    elem_id = int(parts[0])
                    conn = list(map(int, parts[1:]))
                    elements.append([elem_id] + conn)
                
    return np.array(nodes), np.array(elements)


def plot_mesh_with_Uz(nodes, elements, Uz_all,
                      xlim=None, ylim=None, zlim=None):
    """
    Plots 3D mesh colored by Uz with optional spatial zooming.

    Parameters:
    -----------
    nodes : (N,4) → [id, x, y, z]
    elements : (M,9) → [elem_id, n1, ..., n8]
    Uz_all : (N,) → Uz displacements per node
    xlim, ylim, zlim : tuple (min, max) or None
    
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=210)

    # ---- OPTIONAL SPATIAL FILTERING ----
    mask = np.ones(len(nodes), dtype=bool)
    if xlim is not None:
        mask &= (nodes[:,1] >= xlim[0]) & (nodes[:,1] <= xlim[1])
    if ylim is not None:
        mask &= (nodes[:,2] >= ylim[0]) & (nodes[:,2] <= ylim[1])
    if zlim is not None:
        mask &= (nodes[:,3] >= zlim[0]) & (nodes[:,3] <= zlim[1])

    filtered_nodes = nodes[mask]
    filtered_Uz = Uz_all[mask]

    # For element drawing, we keep only elements whose ALL nodes remain in filter
    filtered_node_ids = set(filtered_nodes[:,0].astype(int))
    filtered_elements = []
    for elem in elements:
        node_ids = elem[1:]
        if all(nid in filtered_node_ids for nid in node_ids):
            filtered_elements.append(elem)
    
    # ---- MESH EDGES ----
    edges = []
    hex_edges_idx = [
        (0,1),(1,2),(2,3),(3,0),      # bottom face
        (4,5),(5,6),(6,7),(7,4),      # top face
        (0,4),(1,5),(2,6),(3,7)       # vertical edges
    ]
    for elem in filtered_elements:
        node_ids = elem[1:]
        coords = filtered_nodes[[np.where(filtered_nodes[:,0] == nid)[0][0] for nid in node_ids],1:4]
        for i,j in hex_edges_idx:
            edges.append([coords[i], coords[j]])

    

    # ---- SCATTER UZ ----
    sc = ax.scatter(filtered_nodes[:,1], filtered_nodes[:,2], filtered_nodes[:,3],
                    c=filtered_Uz, cmap='plasma_r', s=8)

    fig.colorbar(sc, shrink=0.5, label='Uz')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3D Mesh (Zoomed)' if (xlim or ylim or zlim) else '3D Mesh')

    ax.auto_scale_xyz(filtered_nodes[:,1], filtered_nodes[:,2], filtered_nodes[:,3])
    plt.show()

    print(f"Nodes shown: {len(filtered_nodes)}, Elements shown: {len(filtered_elements)}")
    
    
def extract_significant_z_planes(nodes, min_count_ratio=0.02):
    """
    Finds Z-planes with many nodes.
    
    Parameters
    ----------
    nodes : array of shape (N,4) or (N,3)
        Node coordinates (col 3 must be z)
    min_count_ratio : float
        Threshold relative to max population (default 10%)
    
    Returns
    -------
    significant_planes : list of float
        Sorted Z-levels with many nodes
    stats : dict
        {z_level: count} for all z planes
    """
    # Extract Z column (works for [id,x,y,z] or [x,y,z])
    if nodes.shape[1] == 4:
        z = nodes[:,3]
    else:
        z = nodes[:,2]

    # Group and count
    unique_z, counts = np.unique(z, return_counts=True)
    
    # threshold based on population
    max_count = counts.max()
    threshold = max_count * min_count_ratio

    # keep significant planes only
    significant_planes = unique_z[counts >= threshold]

    # return sorted
    return list(np.sort(significant_planes)), dict(zip(unique_z, counts))


def get_linear_cut(nodes, Uz_all, fixed_axis='x', fixed_value=0.0, z_plane='min', tol=1e-2):
    """
    Extract a linear cut of Uz along a fixed axis at a given plane.

    Parameters:
    -----------
    nodes : np.ndarray of shape (N,4) -> [id, x, y, z]
    Uz_all : np.ndarray of shape (N,) -> Uz displacement per node
    fixed_axis : str, 'x' or 'y' -> axis to fix
    fixed_value : float -> value of the fixed axis
    z_plane : 'min', 'max', or float -> z coordinate of plane
    tol : float -> tolerance for including nodes near the plane/fixed axis

    Returns:
    --------
    cut_coords : np.ndarray of shape (M,3) -> [x, y, z] of nodes along the line
    cut_Uz : np.ndarray of shape (M,) -> Uz displacement along the line
    line_axis : str -> axis along which the line varies ('x' or 'y')
    """
    # Map axes to column indices
    axis_idx = {'x':1, 'y':2, 'z':3}

    # Determine z plane value
    z_idx = axis_idx['z']
    if z_plane == 'min':
        z_val = np.min(nodes[:,z_idx])
    elif z_plane == 'max':
        z_val = np.max(nodes[:,z_idx])
    else:
        z_val = float(z_plane)

    # Mask nodes near the plane
    mask = np.abs(nodes[:,z_idx] - z_val) < tol

    # Mask nodes near the fixed axis
    mask &= np.abs(nodes[:,axis_idx[fixed_axis]] - fixed_value) < tol

    # Select nodes and Uz
    cut_nodes = nodes[mask]
    cut_Uz = Uz_all[mask]

    if cut_nodes.shape[0] == 0:
        print("No nodes found for this cut! Try increasing 'tol' or adjusting fixed_value.")
        return np.array([]), np.array([]), None

    # Determine the line axis (the one that varies)
    line_axis = 'y' if fixed_axis == 'x' else 'x'
    line_idx = axis_idx[line_axis]

    # Sort by the varying axis
    sorted_idx = np.argsort(cut_nodes[:,line_idx])
    cut_nodes = cut_nodes[sorted_idx]
    cut_Uz = cut_Uz[sorted_idx]

    # Return coordinates of nodes (x, y, z) and Uz
    cut_coords = cut_nodes[:,1:4]

    return cut_coords, cut_Uz, line_axis