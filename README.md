# Material parameter estimation enabled by adaptive deformable body simulation of continuum beam models and real-time visual tracking

We propose a simulation framework that integrates real-time visual tracking with continuum beam models to estimate material parameters accurately. By combining observed motion trajectories with finite element simulations based on Euler-Bernoulli and Timoshenko theories, the method adaptively adjusts material properties—such as stiffness—via optimization to minimize discrepancies between measured and simulated responses. The approach demonstrates high accuracy and computational efficiency, making it suitable for real-time structural monitoring and digital twin applications.

---

## Download Dataset
Download data.zip from this [Google Drive](https://drive.google.com/file/d/1JXhTpRsDGlvDu2qU6Q8SPhkaqf2Grh9N/view?usp=drive_link). Unzip it to the root dir.
The structure should look like the following:
```bash
├── data 
│   ├── Ecoflex0020
│   ├── ...
│   └── PDMS
```

---

## Installation

```bash
git https://github.com/SookmyungHumanAI/real2sim_simple_beam.git
conda env create -f sim_beam.yaml
conda activate sim_beam
```

## Run Optimization
```bash
python main.py --config_root ./ --beam_model timoshenko --time_integration newmark --optimize True --optim_num 100
```

---
## Visualization
### Optimization Result
![disp](https://github.com/user-attachments/assets/db8ea1e0-9cd3-4567-b80f-54fae20c4aa0)

### Stress/ Strain
![stress_strain](https://github.com/user-attachments/assets/5f512d3d-ce08-420a-aaf2-306187ce2df4)


---
## Citation

If you use this code or parts of this simulation framework in your research, please cite the following paper:

```bibtex
@article{lee2025material,
  title={Material parameter estimation enabled by adaptive deformable body simulation of continuum beam models and real-time visual tracking},
  author={Lee, Seongbeen and Park, Yewon and Kim, Suin and Sim, Joo Yong},
  journal={European Journal of Mechanics-A/Solids},
  volume={112},
  pages={105636},
  year={2025},
  publisher={Elsevier}
}
