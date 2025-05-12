import torch

def assemble_K_eb(num_ele, num_dof, E, I, L, nu=None):
    """
    Assemble the global stiffness matrix for a Euler-Bernoulli beam element system.
    - Uses classical beam theory (no shear deformation considered)
    - Constructs a local 4x4 stiffness matrix per element based on material and geometry
    - Assembles the global matrix K by summing element contributions
    - Inputs:
        num_ele : number of beam elements
        num_dof : total degrees of freedom (= 2 * (num_ele + 1))
        E       : Young's modulus
        I       : second moment of area
        L       : length of one element
        nu      : (unused, kept for API consistency)
    - Returns:
        K       : global stiffness matrix (num_dof x num_dof)
    """    
    K = torch.zeros((num_dof, num_dof))# stiffness matrix
    k = E*I/L**3*torch.tensor([[12,    6*L,   -12,   6*L    ],
                                [6*L,  4*L**2, -6*L, 2*L**2],
                                [-12, -6*L,    12,  -6*L   ],
                                [6*L,  2*L**2,-6*L, 4*L**2]])
    for i in range(num_ele):
        K[2*i:2*(i+2), 2*i:2*(i+2)] = K[2*i:2*(i+2), 2*i:2*(i+2)] + k
    
    return K

def assemble_M_eb(num_ele, num_dof, rho, A, L):
    """
    Assemble the global mass matrix for a Euler-Bernoulli beam element system.
    - Constructs the consistent mass matrix for each beam element
    - Based on standard 4x4 mass formulation (no rotary inertia)
    - Assembles the global matrix M by adding local element matrices
    - Inputs:
        num_ele : number of elements
        num_dof : total degrees of freedom
        rho     : material density
        A       : cross-sectional area
        L       : length of one element
    - Returns:
        M       : global mass matrix (num_dof x num_dof)
    """
    M = torch.zeros((num_dof, num_dof))# mass matrix
    m = rho*A*L/420*torch.tensor([[156,   22*L,     54,   -13*L],
                                  [ 22*L,  4*L**2,  13*L, -3*L**2],
                                  [ 54,   13*L,    156,   -22*L],
                                  [-13*L, -3*L**2, -22*L, 4*L**2],])
    for i in range(num_ele):
        M[2*i:2*(i+2), 2*i:2*(i+2)] = M[2*i:2*(i+2), 2*i:2*(i+2)] + m
    
    return M
