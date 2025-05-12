import torch
from torch.nn import MSELoss, Parameter
from torch.optim import Adam

def cdm(in_pos, in_accel, num_ele, delta_t, num_steps, num_dof, I, A, L, d, rho, E, assemble_K, assemble_M, optim=False, nu=None):
    
    K = assemble_K(num_ele, num_dof, E, I, L, d, nu)
    M = assemble_M(num_ele, num_dof, rho, A, L, I)

    M_for_known = torch.zeros((num_dof, num_dof))
    M_for_known[:, :2] = M[:, :2]
    for i in range(2, num_dof):
        M_for_known[i, i] = -(delta_t**2)
    M_for_unknown = torch.zeros((num_dof, num_dof))
    M_for_unknown[0, 0] = -(delta_t**2)
    M_for_unknown[1, 1] = -(delta_t**2)
    M_for_unknown[:, 2:] = M[:, 2:]

    out_disp = torch.zeros((num_dof, num_steps + 1))
    F_fix = torch.zeros((2, num_steps+1))
    f_ext = torch.zeros((num_dof, 1))

    for i in range(2, num_steps):
        t = i * delta_t
        f_ext[:1] = in_pos[i:i+1]
        x = torch.linalg.inv(M_for_unknown) @ \
            (2 * M @ out_disp[:, [i-1]] - M @ out_disp[:, [i-2]] - K @
            out_disp[:, [i-1]]*(delta_t**2) - M_for_known@f_ext)
        out_disp[2:, [i]] = x[2:]
        out_disp[:1, [i]] = in_pos[i:i+1]
        F_fix[:, [i]] = x[:2]
    if optim:
        return out_disp, E, nu
    else:
        return out_disp

def opt_cdm(num_ele, delta_t, num_steps, num_dof, I, A, L, d, rho, init_E, init_lr, \
    base_pos, tip_pos, base_accel, assemble_K, assemble_M, optim_num = 100, nu=-100):
        
    E = Parameter(torch.tensor([init_E], dtype=torch.float))
    if nu != -100:
        nu = Parameter(torch.tensor([nu], dtype=torch.float))
        optimizer = Adam([
                {'params': E,   'lr': init_lr},
                {'params': nu,  'lr': 5e-3}
            ])
        cand_k_s = []
        cand_nu = []
    else:
        optimizer = Adam([E], lr=init_lr)
        
    loss_fn = MSELoss()
    losses = []
    cand_E = []

    for j in range(optim_num):
            cand_E.append(E.item())
            
            if nu != -100:
                cand_k_s.append((6*(1 + nu.item())) / (7 + 6 * nu.item()))
                cand_nu.append(nu.item())
            
            optimizer.zero_grad()
            disp, E, nu = cdm(base_pos, base_accel, num_ele, delta_t, num_steps, num_dof, \
                                I, A, L, d, rho, E, assemble_K, assemble_M, optim=True, nu=nu)
            
            loss = loss_fn(tip_pos, disp[num_dof-2, :-1])
            loss.backward()
            optimizer.step()
            if nu != -100:
                with torch.no_grad():
                    nu.clamp_(0.45+1e-7, 0.5-1e-7) 
            losses.append(torch.sqrt(loss).item())
            
    if nu != -100:
        return disp, cand_E, losses, cand_k_s, cand_nu
    else:
        return disp, cand_E, losses, None, None