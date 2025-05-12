import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.assume_disp import *

def plot_strain_stress(cfg, num_steps, disp, E,  v_func, beam_stress, nu, I, key, save_path=""):
    num_ele = cfg["num_ele"]
    d = cfg["d"]
    l = cfg["l"]
    L = l / num_ele
    end_time = 6

    strain_total = np.empty(0)
    stress_total = np.empty(0)
    curvature_G_total = np.empty(0)
    curvature_EB_total = np.empty(0)
    diff_total = np.empty(0)
    alpha_total = np.empty(0)
    for idx in range(num_steps):
        coeffs = beam_stress(idx, disp, num_ele, L, v_func, cal_v, E, nu, d, I)
        strain_t = cal_strain(coeffs, num_ele, L)
        stress = cal_stress(E, strain_t,d)
        curvature_G, curvature_EB, diff, alpha = cal_curvature(coeffs, num_ele, L)
        if idx==0:
            strain_total = strain_t[:,np.newaxis]
            stress_total = stress[:,np.newaxis]
            curvature_G_total = curvature_G[:, np.newaxis]
            curvature_EB_total = curvature_EB[:, np.newaxis]
            diff_total = diff[:, np.newaxis]
            alpha_total = alpha[:, np.newaxis]
        else:
            strain_total = np.concatenate((strain_total, strain_t[:,np.newaxis]), -1)
            stress_total = np.concatenate((stress_total, stress[:,np.newaxis]), -1)
            curvature_G_total = np.concatenate((curvature_G_total, curvature_G[:, np.newaxis]), -1)
            curvature_EB_total = np.concatenate((curvature_EB_total, curvature_EB[:, np.newaxis]), -1)
            diff_total = np.concatenate((diff_total, diff[:, np.newaxis]), -1)
            alpha_total = np.concatenate((alpha_total, alpha[:, np.newaxis]), -1)

    resized_strain = cv2.resize(strain_total[::-1,:], (400, 300))
    resized_stress = cv2.resize(stress_total[::-1,:]/10, (400, 300)) # g/(cm*s)^2 -> Pa
    cmap = "jet"

    interval = 3
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    
    plt.title("Strain")
    plt.imshow(resized_strain, cmap = cmap)
    plt.ylabel("Beam Length [mm]")
    plt.xlabel("Time [s]")
    plt.xticks(np.linspace(0, 400, interval+1), \
            [f"{t:.0f}" for t in np.linspace(0,end_time, interval+1)])
    plt.yticks(np.linspace(0, resized_strain.shape[0], interval+1), \
            [f"{t*10:.0f}" for t in np.linspace(0,l, interval+1)[::-1]])
    clb = plt.colorbar(shrink=0.8)
    clb.set_label('Strain [mm/mm]', ha='center')
    plt.tight_layout()

    plt.subplot(122)
    plt.title("Stress")
    plt.imshow(resized_stress, cmap = cmap)
    plt.ylabel("Beam Length [mm]")
    plt.xlabel("Time [s]")
    plt.xticks(np.linspace(0, 400, interval+1), \
            [f"{t:.0f}" for t in np.linspace(0,end_time, interval+1)], )
    plt.yticks(np.linspace(0, resized_stress.shape[0], interval+1), \
            [f"{t*10:.0f}" for t in np.linspace(0,l, interval+1)[::-1]], )
    clb = plt.colorbar(shrink=0.8)
    clb.set_label('Stress [N/m^2]', ha="center")
    plt.tight_layout()
    plt.savefig(f"{save_path}/{key}.png")
    plt.show()
    plt.close()
    