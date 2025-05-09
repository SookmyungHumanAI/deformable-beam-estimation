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
    
    return num_ele, mass, l, d, dt, end, start, delta_t, num_steps, num_dof, I, A, L, rho, init_E, init_lr


# def read_camera_data(data_path, marker_name, target_marker, fps, dt, \
#                      plot=False, csv_header=0, smoothing=False, start=9., end=15,\
#                     num_node=3):
#     traj_df = pd.read_csv(data_path, header=csv_header)
#     num_node = num_node + 1
#     h_rate, w_rate = 413.5/1080*12.9, 86.5/1920*2.5
#     points = traj_df[['x', 'y']].to_numpy()
#     points[:,0] = points[:,0]*w_rate
#     points[:,1] = points[:,1]*h_rate
#     points = points.reshape(-1,num_node,2)
#     if end==-1:
#         end = len(points)/60
#     duration = points.shape[0]/60
#     points = points[int(start*60):int(end*60)]
#     num_frame = int(end*60) - int(start*60)
#     print(points.shape[0], points.shape[0]/60,"sec")
#     position_np = points[:,:,0] - points[0,:,0]
#     time_original = np.linspace(0, len(position_np) * fps, len(position_np))    
#     time_new = np.linspace(0, len(position_np) * fps, int(len(position_np)*fps/dt))
#     positions = np.empty(0)
#     for i in range(num_node)[::-1]:
#         pos = interp1d(time_original, position_np[:,i], kind='linear')(time_new)
#         plt.show()
#         positions = np.concatenate((positions, pos))
#     positions = positions.reshape(num_node, -1).transpose(1,0)
    
#     # for num_plot, idx in enumerate(range(num_node)):
#     #     plt.plot(time_original,points[:,idx,0] - points[0,idx,0], label=f"{idx:d}")
#     #     plt.plot(time_new,positions[:,idx] - positions[0,idx], label=f"{idx:d}")
#     #     plt.legend()
    
#     print(positions.shape, time_new.shape)
#     if smoothing:
#         print("****************filtered!!****************")
#         filter_num = 40
#         filter=1*filter_num
        
#         ft_poss = np.empty(0)
#         ft_vels = np.empty(0)
#         ft_accs = np.empty(0)
#         poss = np.empty(0)
#         vels = np.empty(0)
#         accs = np.empty(0)
        
#         for idx in range(num_node):
#             b, a = butter(N=3, Wn=filter/60, btype='low')
#             poss = np.concatenate((poss, positions[:,idx]))
#             ft_pos = filtfilt(b, a, positions[:,idx])
#             ft_poss = np.concatenate((ft_poss, ft_pos))
            
#             vel = np.diff(ft_pos) / dt
#             vels = np.concatenate((vels, vel))
#             ft_vel = filtfilt(b, a, vel)
#             ft_vels = np.concatenate((ft_vels, ft_vel))
            
#             b, a = butter(N=3, Wn=filter/60, btype='low')
#             acc = np.diff(ft_vel) / dt
#             accs = np.concatenate((accs, acc))
#             ft_acc = filtfilt(b, a, acc)
#             ft_accs = np.concatenate((ft_accs, ft_acc))
        
#         ft_poss = ft_poss.reshape(num_node, -1).transpose(1,0)
#         ft_vels = ft_vels.reshape(num_node, -1).transpose(1,0)
#         ft_accs = ft_accs.reshape(num_node, -1).transpose(1,0)
        
#         poss = poss.reshape(num_node, -1).transpose(1,0)
#         vels = vels.reshape(num_node, -1).transpose(1,0)
#         accs = accs.reshape(num_node, -1).transpose(1,0)
        
#     else:
#         ft_poss = np.empty(0)
#         ft_vels = np.empty(0)
#         ft_accs = np.empty(0)
#         poss = np.empty(0)
#         vels = np.empty(0)
#         accs = np.empty(0)
        
#         for idx in range(num_node):
#             ft_poss = np.concatenate((ft_poss, positions[:,idx]))
#             poss = np.concatenate((poss, positions[:,idx]))
#             vel = np.diff(positions[:,idx]) / dt
#             vels = np.concatenate((vels, vel))
#             ft_vels = np.concatenate((ft_vels, vel))
#             acc = np.diff(vel) / dt
#             accs = np.concatenate((accs, acc))
#             ft_accs = np.concatenate((ft_accs, acc))
        
#         ft_poss = ft_poss.reshape(num_node, -1).transpose(1,0)
#         ft_vels = ft_vels.reshape(num_node, -1).transpose(1,0)
#         ft_accs = ft_accs.reshape(num_node, -1).transpose(1,0)
        
#         poss = poss.reshape(num_node, -1).transpose(1,0)
#         vels = vels.reshape(num_node, -1).transpose(1,0)
#         accs = accs.reshape(num_node, -1).transpose(1,0)
#     if plot:
#         plt.figure(figsize=(10,6))
#         for idx in range(num_node)[::-1]:
#             plt.subplot(3,1,1)
#             plt.title("Position")
#             plt.plot(time_new, poss[:,idx], linewidth=0.8)
#             plt.plot(time_new, ft_poss[:,idx], linewidth=0.8)
#             plt.subplot(3,1,2)
#             plt.title("Velocity")
#             plt.plot(time_new[1:], vels[:,idx], linewidth=0.8)
#             plt.plot(time_new[1:], ft_vels[:,idx], linewidth=0.8)
#             plt.subplot(3,1,3)
#             plt.title("Acceleration")
#             plt.plot(time_new[2:], accs[:,idx], linewidth=0.8)
#             plt.plot(time_new[2:], ft_accs[:,idx], linewidth=0.8)
#             break
#         plt.tight_layout()
#         plt.show()
#         plt.close()        
#     return ft_poss, ft_vels, ft_accs




############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
# def beam_positions(displacements, num_ele):
#     positions = np.zeros((num_ele + 1, len(displacements[0])))
#     for i in range(num_ele+1):
#         positions[i] = displacements[2*i]
#     return positions

# def vis_beam_displacement(save_path, num_dof, disp=None, disp_path=None, target=None):
#     # if disp == None:
#     #     disp = np.load(disp_path)
#     # assert disp!= None or disp_path!=None, print("Give disp tensor or disp_path")
    
#     if os.path.exists(save_path):
#         print("warning! already exists!")
#     else:
#         os.makedirs(save_path)
#     plt.figure(figsize=(10,3))
#     plt.title("Transverse Displacement")
#     for i in range(num_dof//2):
#         plt.plot(disp[2*i], label = f"{2*i}")
#     plt.plot(target, label="target")
#     plt.legend()
#     plt.savefig(f"{save_path}/disp_transverse.svg")
#     plt.close()
#     plt.figure(figsize=(10,3))
#     plt.title("Angular Displacement")
#     for i in range(num_dof//2):
#         plt.plot(disp[2*i], label = f"{2*i}")
#     plt.legend()
#     plt.savefig(f"{save_path}/disp_angular.svg")
#     plt.close()

# def vis_beam_pos(num_ele, l, dt, end_time, save_path, disp=None, disp_path=None):
#     num_steps = int(end_time/dt)
#     L = l/num_ele
#     # if disp == None:
#     #     disp = np.load(disp_path)
#     # assert disp!= None or disp_path!=None, print("Give disp tensor or disp_path")
    
#     disp = np.array(disp)
#     # Calculate the positions of node according to displacement
#     beam_pos = beam_positions(disp, num_ele)

#     # Setting for animation
#     fig, ax = plt.subplots()
#     line, = ax.plot([], [], lw=2)
#     ax.set_xlim(0, num_ele)
#     lim = max(abs(beam_pos.min()), abs(beam_pos.max()))
#     ax.set_ylim(-lim*2, lim*2)
#     ax.grid()
#     ax.set_xticks(np.array(range(num_ele+1)))
#     ax.set_xlabel('Index of Node')
#     ax.set_ylabel('Beam Position [m]')

#     index_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

#     # Update function
#     def animate(i):
#         line.set_data(range(num_ele + 1), beam_pos[:, i])
#         time_stamp = i*dt
#         index_text.set_text(f'{time_stamp:.4f} sec')
#         return (line, index_text)

#     # Generate animation
#     ani = animation.FuncAnimation(fig, animate, frames=range(0, num_steps, 100), interval=33, blit=True)

#     # Save animation
#     ani.save(f"{save_path}/node_position.gif", writer='ffmpeg', fps=60)


# def read_single_data(config_path):
#     # print(config_path)
#     with open(config_path, 'r') as f:
#         cfg = yaml.full_load(f)
#     sample_name = config_path.split("/")[-1].split(".")[0].split("\\")[-1]
#     num_dof, I, A, L, rho = cylinder(cfg)
#     axis = cfg["axis"]
#     duration = 6
#     cfg["sim_start"] = cfg["sim_start"]# + 0.5
#     start = cfg["sim_start"]
#     cfg["sim_end"] = cfg["sim_end"]#+ 0.5
#     end = cfg["sim_end"]
#     cfg["dt"]=cfg["dt"]
#     dt = cfg["dt"]
#     filter_num = cfg["filter"]
#     if config_path.split("/")[2].startswith("240529"):
#         fps=1/120
#     elif config_path.split("/")[2].startswith("240605") or config_path.split("/")[2].startswith("240309"):
#         fps=1/500

#     plot = False
#     pos, vel, accel = read_optitrack(cfg, cfg["rigid_marker"], fps=fps, plot=plot, \
#                             csv_header=0, start=start, end=end, axis=axis, filter_num=filter_num)
#     tip_pos, tip_vel, tip_accel = read_optitrack(cfg, cfg["tip_marker"], fps=fps, plot=plot, \
#                             csv_header=0, start=start, end=end, axis=axis, filter_num=filter_num)

#     max_acc = max(abs(accel))
#     pos = pos - pos[0]
#     tip_pos = tip_pos - tip_pos[0]
#     pos = torch.tensor(pos.copy(), dtype=torch.float)*100 # [cm]
#     tip_pos = torch.tensor(tip_pos.copy(), dtype=torch.float)*100 # [cm]
#     num_steps = min(len(pos), int(duration/dt))

#     tip_accel = torch.tensor(tip_accel.copy(), dtype=torch.float)*100
#     return cfg, A, axis, dt, duration, end, filter_num, fps, I, L, max_acc, num_dof, num_steps, pos, rho, start, tip_accel, tip_pos, tip_vel, vel, pos, vel, accel, sample_name

# def read_selected(all_data=False):
#     if all_data:
#         materials = nat(glob("../materials/240529/*/*.yaml")+glob("../materials/240605/*/*.yaml"))
#         selected_list = [path.replace(os.sep, "/") for path in materials]
#     else:
#         selected_list = []
#         f = open("../trash4/selected.txt", 'r')
#         while True:
#             line = f.readline()
#             if not line: break
#             selected_list.append(line[:-1])
#         f.close()
#     print(f"Append {len(selected_list)} files to list")
#     return selected_list