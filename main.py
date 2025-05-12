import argparse
import numpy as np
import os
import torch
import yaml
from datetime import datetime
from natsort import natsorted as nat
from glob import glob

from beam import euler_bernoulli, timoshenko
from utils import assume_disp, src, beam_summary_plots, beam_visualization
from time_integ import cdm, newmark


class DeformableBeamSim:
    def __init__(self, config_paths, beam_model, time_integration, optimize, optim_num=0):
        self.assemble_M = None
        self.assemble_K = None
        self.time_int = None
        
        self.config_paths = nat(glob(f"{config_paths}/data/*/*.yaml"))
        self.beam_model = beam_model
        self.time_integration = time_integration
        self.sim_idx = 0
        self.optimize = optimize
        if self.optimize:
            assert optim_num > 0, "Define the number of optimization iteration"
            self.optim_num = optim_num
        
        # save data
        self.cand_E = None
        self.losses = None
        self.cand_k_s = None
        self.cand_nu = None
        self.vis_data_disp = False
        self.vis_result = True
        self.configs = {}
        self.results = {}
        self.save_path = None
        if not os.path.exists("./results"):
            os.mkdir("./results")
        self.set()
        
    def set(self):
        beam_models = {
            "euler-bernoulli": (euler_bernoulli.assemble_M_eb, euler_bernoulli.assemble_K_eb, assume_disp.eb_v_func),
            "timoshenko": (timoshenko.assemble_M_timo, timoshenko.assemble_K_timo, assume_disp.tm_v_func)
        }
        try:
            self.assemble_M, self.assemble_K, self.v_func = beam_models[self.beam_model]
        except KeyError:
            raise ValueError("Wrong Beam Model")
            
        time_int_map = {
            "cdm": (cdm.cdm, cdm.opt_cdm),
            "newmark": (newmark.newmark, newmark.opt_newmark)
        }

        try:
            self.time_int = time_int_map[self.time_integration][1 if self.optimize else 0]
        except KeyError:
            raise ValueError("Wrong Time Integration Method")
        
        print(f"[INFO] Initialized with beam model: {self.beam_model}, time integration: {self.time_integration}, optimize: {self.optimize}")

    
    def load_config(self):
        self.config_path = self.config_paths[self.sim_idx]
        with open(self.config_path, 'r') as f:
            self.cfg = yaml.full_load(f)
        self.key = self.cfg["save_fname"]
        
    def prepare_data(self):
        tip_pos, tip_accel = src.read_optitrack(self.config_path, "tip", plot=self.vis_data_disp)
        base_pos, base_accel = src.read_optitrack(self.config_path, "base", plot=self.vis_data_disp)
        
        self.num_ele, self.dt, self.delta_t, self.num_steps, self.num_dof, self.I, self.A, self.L, self.rho, self.init_E, self.init_lr = \
            src.init_simulation(self.config_path)
            
        self.base_pos = torch.tensor(base_pos, dtype=torch.float)*100
        self.tip_pos = torch.tensor(tip_pos, dtype=torch.float)*100
        self.base_accel_shape = base_accel.shape
        self.num_steps = min(len(base_pos), int(6/self.dt))
        
    def run_optimization(self):
        *_, self.cand_E, self.losses, self.cand_k_s, self.cand_nu = \
                self.time_int(self.num_ele, self.delta_t, self.num_steps, self.num_dof, self.I, self.A, self.L, self.rho, self.init_E, self.init_lr, self.base_pos, self.tip_pos, torch.zeros(self.base_accel_shape), self.assemble_K, self.assemble_M, optim_num = self.optim_num, nu=0.35)
        self.disp = newmark.newmark(self.base_pos, torch.zeros(self.base_accel_shape), \
                            self.num_ele, self.delta_t, self.num_steps, self.num_dof, \
                            self.I, self.A, self.L, self.rho, self.cand_E[0], \
                            self.assemble_K, self.assemble_M, optim=False)
        self.init_disp = newmark.newmark(self.base_pos, torch.zeros(self.base_accel_shape), \
                            self.num_ele, self.delta_t, self.num_steps, self.num_dof, \
                            self.I, self.A, self.L, self.rho, self.cand_E[np.argmin(self.losses)], \
                            self.assemble_K, self.assemble_M, optim=False)
        
    def run_simulation(self):
        self.disp = self.time_int(self.base_pos, torch.zeros(self.base_accel_shape), \
                                self.num_ele, self.delta_t, self.num_steps, self.num_dof, \
                                self.I, self.A, self.L, self.rho, self.init_E, \
                                self.assemble_K, self.assemble_M, optim=False)
    def save_result(self):
        key = self.key
        self.configs[key] = self.cfg
        self.results[key] = {}
        self.results[key]["disp"] = self.disp
        if self.optimize:
            self.results[key]["cand_E"] = self.cand_E
            self.results[key]["losses"] = self.losses
            if self.cand_k_s is not None and self.cand_nu is not None:
                self.results[key]["cand_k_s"] = self.cand_k_s
                self.results[key]["cand_nu"] = self.cand_nu
            
        
    def sim(self):
        self.load_config()
        self.prepare_data()
        
        if self.optimize:
            self.run_optimization()
        else:
            self.run_simulation()
        
        self.save_result()
        
        self.sim_idx += 1

    def visualize_result(self):
        if self.save_path == None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_path = f"./results/{now}"
        result_path = f"{self.save_path}/result"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        print(f"[INFO] Visualizing results to: {self.save_path}/result")
        if self.optimize:
            beam_summary_plots.plot_total_optimize(self.num_dof, self.num_steps, self.disp, self.init_disp, self.tip_pos, self.base_pos, save_path=result_path, save_fname=self.key, losses=self.losses, cand_E=self.cand_E)
        else:
            beam_summary_plots.plot_total_sim(self.num_dof, self.disp, self.tip_pos, self.base_pos, save_path=result_path, save_fname=self.key)
        
    def visualize_stress(self):        
        if self.save_path == None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_path = f"./results/{now}"
        stress_path = f"{self.save_path}/stress"
        if not os.path.exists(stress_path):
            os.makedirs(stress_path)
        print(f"[INFO] Visualizing results to: {stress_path}")
        
        E = self.cand_E[np.argmin(self.losses)] if self.optimize else self.init_E
        nu = self.cand_nu[np.argmin(self.losses)]
        beam_visualization.plot_strain_stress(self.cfg, self.num_steps, self.disp, E, self.v_func, nu, self.I, self.key, save_path=stress_path)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deformable beam simulation.")
    parser.add_argument("--config_root", type=str, default="./",
                        help="Path to the root directory containing config YAMLs (default: ./configs)")
    parser.add_argument("--beam_model", type=str, choices=["euler-bernoulli", "timoshenko"], default="timoshenko",
                        help="Beam model to use")
    parser.add_argument("--time_integration", type=str, choices=["cdm", "newmark"], default="newmark",
                        help="Time integration method")
    parser.add_argument("--optimize", action="store_true",
                        help="Enable optimization of material parameters")
    parser.add_argument("--optim_num", type=int, default=100,
                        help="Number of optimization iterations (used only if --optimize is set)")

    args = parser.parse_args()

    # Initialize simulation
    sim = DeformableBeamSim(
        config_paths=args.config_root,
        beam_model=args.beam_model,
        time_integration=args.time_integration,
        optimize=args.optimize,
        optim_num=args.optim_num
    )

    # Run all simulations
    num_cases = len(sim.config_paths)
    print(f"[INFO] Running {num_cases} simulation case(s)")
    for _ in range(num_cases):
        sim.sim()
        if sim.vis_result:
            sim.visualize_result()
            sim.visualize_stress()