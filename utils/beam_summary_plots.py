import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.assume_disp import *
        
def plot_total_optimize(num_dof, num_steps, opt_disp, init_disp, tip_pos, base_pos, save_path="", save_fname="", losses=None, cand_E=None):
    end_time = 6
    k=2*(num_dof//2-1)
    
    # Create subplots
    fig = plt.figure(figsize=(8, 7.5))

    # Add subplots with the desired layout
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((3, 2), (2, 0))
    ax4 = plt.subplot2grid((3, 2), (2, 1))

    ax1.plot(np.linspace(0, end_time, len(tip_pos)),tip_pos, \
        label = "Ground Truth")
    ax1.plot(np.linspace(0, end_time, len(tip_pos)),(init_disp[k,:-1].detach()), \
        label = "FEM with E_init")
    ax1.plot(np.linspace(0, end_time, num_steps), opt_disp[k,:-1].detach(), \
        label = f"FEM with E_opt")
    ax1.set_title('Transverse Displacement')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Displacement [cm]')
    ax1.legend()

    ax2.plot(np.linspace(0, end_time, len(tip_pos)),tip_pos-base_pos, \
        label = "Ground Truth")
    ax2.plot(np.linspace(0, end_time, len(tip_pos)),(init_disp[k,:-1].detach())-base_pos, \
        label = "FEM with E_init")
    ax2.plot(np.linspace(0, end_time, num_steps), opt_disp[k,:-1].detach()-base_pos, \
        label = f"FEM with E_opt")
    ax2.set_title('Relative Transverse Displacement')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Relative Displacement [cm]')
    ax2.legend()

    ax3.plot(losses, ".-", label = "losses")
    ax3.set_title('Losses')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('RMSE Loss [cm]')
    ax3.legend()

    ax4.plot(np.array(cand_E)/10, ".-", label="optim_E")
    ax4.set_title('Optimized Elastic Modulus')
    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('Elastic Modulus [Pa]')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}/{save_fname}.png")
    plt.show()
    
def plot_total_sim(num_dof, disp, tip_pos, base_pos, save_path="", save_fname="",):
    end_time = 6
    k=2*(num_dof//2-1)
    
    # Create subplots
    plt.figure(figsize=(8, 5))
    plt.subplot(211)
    plt.plot(np.linspace(0, end_time, len(tip_pos)),tip_pos, \
        label = "Ground Truth")
    plt.plot(np.linspace(0, end_time, len(tip_pos)),(disp[k,:-1].detach()), \
        label = "FEM with E_given")
    plt.title('Transverse Displacement')
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [cm]')
    plt.legend()

    plt.subplot(212)
    plt.plot(np.linspace(0, end_time, len(tip_pos)),tip_pos-base_pos, \
        label = "Ground Truth")
    plt.plot(np.linspace(0, end_time, len(tip_pos)),(disp[k,:-1].detach())-base_pos, \
        label = "FEM with E_given")
    plt.title('Relative Transverse Displacement')
    plt.xlabel('Time [s]')
    plt.ylabel('Relative Displacement [cm]')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}/{save_fname}.png")
    plt.show()