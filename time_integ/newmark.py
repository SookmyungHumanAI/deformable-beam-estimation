import torch
from torch.nn import MSELoss, Parameter
from torch.optim import Adam

def F(t, num_ele, L, accel, A, rho):
    ret = torch.tensor([[0], [0]])
    w = rho * accel[t] * A
    for i in range(num_ele):
        if i==num_ele-1:
            ret = torch.cat((ret, torch.tensor([[-w*L/2], [w*L**2/12]])))
        else:
            ret = torch.cat((ret, torch.tensor([[-w*L], [0]])))
    return ret

def newmark(in_pos, in_accel, num_ele, delta_t, num_steps, num_dof, I, A, L, d, rho, E, assemble_K, assemble_M, optim=False, nu=None):
    beta = 1/4
    gamma = 1/2
    
    K = assemble_K(num_ele, num_dof, E, I, L, d, nu)
    M = assemble_M(num_ele, num_dof, rho, A, L, I)
    
    K_new = torch.linalg.inv((K + 1/(beta*delta_t**2)*M))
    F_0 = F(t=0, num_ele=num_ele, L=L, accel=in_accel, A=A, rho=rho)[2:]
    out_disp = torch.zeros((num_dof, num_steps + 1))
    f_ext = torch.zeros((num_dof,1))

    M_for_unknown = torch.eye(num_dof)
    M_for_unknown[:,:2] = K_new[:,:2]
    M_for_unknown_inv = torch.linalg.inv(M_for_unknown)

    out_vel = torch.zeros_like(out_disp)
    out_acc = torch.zeros_like(out_disp)
    out_acc[2:, [0]] = torch.linalg.inv(M[2:, 2:])@(F_0)
    K_new_M = K_new@M/(beta*delta_t**2)
    
    for i in range(1, num_steps):
        f_ext[:1,:] = in_pos[i:i+1]
        x = M_for_unknown_inv@\
        (- f_ext + K_new_M@(out_disp[:,[i-1]] + delta_t*out_vel[:,[i-1]] + (1/2-beta)*delta_t**2*out_acc[:,[i-1]]))
        out_disp[2:,[i]] = x[2:]
        out_disp[:1,[i]] = in_pos[i:i+1]
        out_acc[2:, [i]] = 1/(beta*delta_t**2)*(out_disp[2:, [i]] - out_disp[2:, [i-1]]\
                                    - delta_t*out_vel[2:, [i-1]] - delta_t**2*(1/2-beta)*out_acc[2:,[i-1]])
        out_vel[2:, [i]] = out_vel[2:, [i-1]] + delta_t*((1-gamma)*out_acc[2:, [i-1]] + gamma*out_acc[2:, [i]])
    if optim:
        return out_disp, E, nu
    else:
        return out_disp

def opt_newmark(num_ele, delta_t, num_steps, num_dof, I, A, L, d, rho, init_E, init_lr, \
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
        disp, E, nu = newmark(base_pos, base_accel, num_ele, delta_t, num_steps, num_dof, \
                                I, A, L, d, rho, E, assemble_K, assemble_M, optim=True, nu=nu)
        loss = loss_fn(tip_pos, disp[num_dof-2,:-1])
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