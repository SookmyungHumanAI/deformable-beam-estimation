import numpy as np

'''  Euler-Bernoulli  '''

def eb_v_func(idx, disp, num_e, L, E, nu, d, I):
    # v(x) = a[0]x^3 + a[1]x^2 + a[2]x^1 + a[3]
    # a[0]: v_1
    # a[1]: phi_1
    # a[2]: v_2 = disp[2*num_e + 0,idx]
    # a[3]: phi_2 = disp[2*num_e + 1,idx]
    a = np.zeros([4])
    a[3] = disp[2*num_e + 0,idx]
    a[2] = disp[2*num_e + 1,idx]
    a[0], a[1] = np.linalg.inv(np.array([[L**3, L**2],[3*L**2, 2*L]]))@\
                np.array([[disp[2*num_e + 2, idx]-a[2]*L-a[3]], [disp[2*num_e + 3, idx]-a[2]]])
    return a

'''  Timoshenko  '''

def tm_phi_func(a, x, g, k_s, G):
    return a[3] + 2*a[2]*x + (3*x^2+6*g + 6/(k_s*a*G))*a[1]

def tm_v_func(idx, disp, num_e, L, E, nu, d, I):
    # v(x) = a[0]x^3 + a[1]x^2 + a[2]x^1 + a[3]
    v1 = disp[2*num_e + 0,idx]
    phi1 = disp[2*num_e + 1,idx]
    v2 = disp[2*num_e + 2,idx]
    phi2 = disp[2*num_e + 3,idx]
    
    G = E / (2 * (1 + nu))
    k_s = (6*(1 + nu)) / (7 + 6 * nu)
    A_s = k_s * (np.pi / 4) * d**2 # shear area
    g = E*I/(A_s*G)
    denom = L*(L**2+12*g)
    
    a = np.zeros([4])
    a[3] = v1
    a[2] = (-12*g*v1 + (L**3 + 6*g*L)*phi1 + 12*g*v2-6*g*L*phi2) / denom
    a[1] = (-3*L*v1 - (2*L**2 + 6*g)*phi1 + 3*L*v2+(-L**2+6*g)*phi2) / denom
    a[0] = (2*v1 + L*phi1-2*v2+L*phi2) / denom
    return a

'''  Euler-Bernoulli & Timoshenko  '''

def cal_v(x, coeff):
    return coeff[0]*x**3 + coeff[1]*x**2 + coeff[2]*x**1 + coeff[3]

def cal_strain(coeffs, num_ele, L):
    total_strain = np.empty(0)
    x = np.linspace(0, L, 100)
    for num_e in range(num_ele):
        a, b, c, d = coeffs[num_e]
        strain = 6*a*x+2*b
        total_strain = np.concatenate((total_strain, strain))
    return total_strain

def cal_stress(E, total_strain, d):
    return E*total_strain*(-d/2)

def beam_stress(idx, disp, num_ele, L, v_func, cal_v, E, nu, d, I):
    disp = disp.detach().numpy()
    coeffs = np.empty(0)
    for num_e in range(num_ele):
        coeff = v_func(idx, disp, num_e, L, E, nu, d, I)
        if num_e==0:
            coeffs = coeff[np.newaxis,:]
        else:
            coeffs = np.concatenate((coeffs, coeff[np.newaxis,:]), 0)
        tranverse = cal_v(np.linspace(0*L,1*L,100), coeff)
        if num_e==0:
            x0 = tranverse[0]
    return coeffs

def cal_curvature(coeffs, num_ele, L):
    total_curvature_G = np.empty(0)
    total_curvature_EB = np.empty(0)
    diff_1_total = np.empty(0)
    total_alpha = np.empty(0)
    x = np.linspace(0, L, 100)
    x = np.array([L/2])
    for num_e in range(num_ele):
        a, b, c, d = coeffs[num_e]
        diff_1 = 3 * a * x**2 + 2 * b * x + c
        diff_2 = 6 * a * x + 2 * b
        kappa_G = diff_2/((1 + diff_1**2)**1.5)
        alpha = (1 + diff_1**2)**1.5
        total_curvature_G = np.concatenate((total_curvature_G, kappa_G))
        total_curvature_EB = np.concatenate((total_curvature_EB, diff_2))
        total_alpha = np.concatenate((total_alpha, alpha))
        diff_1_total = np.concatenate((diff_1_total, diff_1))

    return total_curvature_G, total_curvature_EB, diff_1_total, total_alpha