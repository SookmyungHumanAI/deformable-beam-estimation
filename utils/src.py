import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import pandas as pd
import torch
import yaml
# from beam import cylinder
from glob import glob
from natsort import natsorted as nat
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt



import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

def read_optitrack(cfg_path, target_marker, plot=False, csv_header=0):
    with open(cfg_path, 'r') as f:
        cfg = yaml.full_load(f)
    
    data_path = cfg["load_path"]
    material_name = cfg["material_name"]
    dt = cfg["dt"]
    axis = cfg["axis"]
    start = cfg["sim_start"]
    end = cfg["sim_end"]
    fps = cfg["fps"]
    filter_num = cfg["filter"]
    if target_marker=="base":
        target_marker = cfg["rigid_marker"]
    elif target_marker=="tip":
        target_marker = cfg["tip_marker"]
    
    # Load CSV data and initialize rigid_motion
    data = pd.read_csv(data_path, header=csv_header)
    rigid_motion = dict()
    
    for i in range(data.shape[1]):
        key = data.iloc[1,i]
        if str(key).startswith(material_name) and (str(key) not in ["nan", "Name"]):
            position = data.iloc[4,i]
            series = data.iloc[8:,i].values.astype(np.float64)
            if not key in rigid_motion.keys():
                rigid_motion[key] = dict()
                pass
            rigid_motion[key][position] = series
    
    position_np = rigid_motion[f'{material_name}:{target_marker}'][axis][int(start/fps):int(end/fps)]
    
    # Smoothing
    b, a = butter(N=3, Wn=filter_num*fps, btype='low')
    ft_pos = filtfilt(b, a, position_np)
    vel = np.diff(ft_pos) / fps
    acc = np.diff(vel) / fps
    
    # Interpolation
    time_original = np.linspace(0, len(ft_pos) * fps, len(ft_pos))
    time_new = np.linspace(0, len(ft_pos) * fps, int(len(ft_pos)*fps/dt))
    itp_position = interp1d(time_original, ft_pos, kind='cubic')(time_new)
    itp_acc = interp1d(time_original[2:], acc, kind='cubic')(time_new[time_new>time_original[2]])

    if plot:
        plt.figure(figsize=(8,5))
        plt.subplot(211)
        plt.title(f"Position({target_marker})")
        plt.plot(time_new, itp_position, linewidth=0.8,label="filtered")
        plt.plot(time_original, position_np, linewidth=0.8, label="original")
        plt.xlabel("Time (s)")
        plt.ylabel("Length [m]")
        plt.legend()
        plt.subplot(212)
        plt.title(f"Acceleration({target_marker})")
        plt.plot(time_new[1:], itp_acc, linewidth=0.8, label="filtered")
        plt.plot(time_original[2:], acc, linewidth=0.8, label="original")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration [m/s^2]")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()
        
    return itp_position-itp_position[0], itp_acc

def cylinder(num_ele, mass, d, l):
    '''
    Parameters:
     - num_ele: Number of elements in the beam
     - mass: Mass of the beam [kg]
     - E: Young's modulus of the beam material [Pa]
     - l: Length of the beam [m]
     - a: Width of the cross-sectional area of the beam [m]
     - b: Height of the cross-sectional area of the beam [m]

    Returns:
     - num_dof: Degrees of freedom, including vertical displacement (δ) and rotational displacement (θ)
     - rho: Density of the beam material [kg/m^3]
     - I: Moment of inertia of the beam's cross-section [m^4]
     - A: Area of the cross-section [m^2]
     - L: Length of one element of the beam [m]
    '''  
    
    num_dof = 2 * num_ele + 2
    rho = mass/((d/2)**2*l*np.pi)
    I = (np.pi*(d/2)**4)/4
    A = np.pi*(d/2)**2
    L = l / num_ele
    return num_dof, I, A, L, rho

def init_simulation(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.full_load(f)
    num_ele = int(cfg["num_ele"])
    mass = float(cfg["mass"])
    l = float(cfg["l"])
    d = float(cfg["d"])
    dt = float(cfg["dt"])

    end = float(cfg["sim_end"])
    start = float(cfg["sim_start"])

    delta_t = torch.tensor([dt], dtype=torch.float)
    num_steps = int((end-start)/dt)
    
    num_dof, I, A, L, rho = cylinder(num_ele, mass, d, l)
    
    init_E = cfg["init_E"]
    init_lr = cfg["init_lr"]
    
    return num_ele, dt, delta_t, num_steps, num_dof, I, A, L, rho, init_E, init_lr
