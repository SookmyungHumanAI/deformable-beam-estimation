import torch
from math import pi


def assemble_K_timo(num_ele, num_dof, E, I, L, d, nu = 0.35):
    """
    Assemble the global stiffness matrix for a Timoshenko beam element system.
    - Includes shear deformation via shear correction factor (k_s) and effective shear area (A_s)
    - Computes the local stiffness matrix for each element, accounting for phi = shear flexibility
    - Assembles the global matrix K from local matrices
    - Inputs:
        num_ele : number of beam elements
        num_dof : total degrees of freedom (= 2 * (num_ele + 1))
        E       : Young's modulus
        I       : second moment of area
        L       : length of one element
        d       : diameter (for circular cross-section)
        nu      : Poisson's ratio (default: 0.35)
    - Returns:
        K       : global stiffness matrix (num_dof x num_dof)
    """
    K = torch.zeros((num_dof, num_dof))# stiffness matrix
    G = E / (2 * (1 + nu))
    k_s = (6*(1 + nu)) / (7 + 6 * nu)
    A_s = k_s * (pi / 4) * d**2 # shear area
    # for a solid circular cross section it is taken as 0.9 times the cross section
    # A_s = k_s * A, k_s: shear correction factor
    phi = 12*E*I/(A_s*G*L**2)
    k = (E * I)/((1+phi)*L**3)*torch.tensor([[12,   6*L,            -12,    6*L    ],
                                            [6*L,   (4+phi)*L**2,   -6*L,   (2-phi)*L**2],
                                            [-12,   -6*L,           12,     -6*L   ],
                                            [6*L,   (2-phi)*L**2,   -6*L,   (4+phi)*L**2]])
    for i in range(num_ele):
        K[2*i:2*(i+2), 2*i:2*(i+2)] = K[2*i:2*(i+2), 2*i:2*(i+2)] + k
        
    return K

def assemble_M_timo(num_ele, num_dof, rho, A, L, I):
    """
    Assemble the full global mass matrix for a Timoshenko beam element (including rotary inertia).
    - Computes translational mass matrix (m_t) and rotary inertia matrix (m_r)
    - Adds both to construct the complete local mass matrix for each element
    - Assembles the global mass matrix M from these local contributions
    - Inputs:
        num_ele : number of elements
        num_dof : total degrees of freedom
        rho     : material density [kg/m^3]
        A       : cross-sectional area
        L       : length of one element
        I       : second moment of area
    - Returns:
        M       : global mass matrix (num_dof x num_dof)
    """

    M = torch.zeros((num_dof, num_dof), dtype=torch.float)

    # -----------------------
    # Translational mass
    # -----------------------
    m_t = rho * A * L / 420.0 * torch.tensor([
        [156.0,     22.0*L,     54.0,       -13.0*L  ],
        [22.0*L,    4.0*L**2,   13.0*L,     -3.0*L**2],
        [54.0,      13.0*L,     156.0,      -22.0*L  ],
        [-13.0*L,   -3.0*L**2,  -22.0*L,    4.0*L**2 ]
    ], dtype=torch.float)

    # -----------------------
    # Rotary inertia term
    # -----------------------
    m_r = rho * I / (420.0 * L) * torch.tensor([
        [36.0,    3.0*L,    -36.0,    3.0*L],
        [3.0*L,   4.0*L**2, -3.0*L,  -1.0*L**2],
        [-36.0,   -3.0*L,    36.0,   -3.0*L],
        [3.0*L,  -1.0*L**2, -3.0*L,   4.0*L**2]
    ], dtype=torch.float)

    m_local = m_t + m_r
    
    for i in range(num_ele):
        idx = slice(2*i, 2*i+4)
        M[idx, idx] += m_local

    return M