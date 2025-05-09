import torch

def assemble_K_eb(num_ele, num_dof, E, I, L):
    """
    Assembles the global stiffness matrix for a beam element system.

    Returns:
    - K (torch.Tensor): The global stiffness matrix of size (num_dof, num_dof).

    This function constructs the global stiffness matrix by computing the local stiffness matrix
    for each beam element based on its material properties (E, I) and geometric properties (L),
    and then assembling these local matrices into the global stiffness matrix.
    The stiffness matrix is crucial for solving structural analysis problems
    to determine displacements and forces within the system.
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
    Assembles the global mass matrix for a beam element system.

    Returns:
    - M (torch.Tensor): The global mass matrix of size (num_dof, num_dof).

    This function constructs the global mass matrix by computing the local mass matrix
    for each beam element based on its material density (rho) and geometric properties (A, L),
    and then assembling these local matrices into the global mass matrix.
    The mass matrix is essential for dynamic analysis of structures to understand
    how they will react to dynamic loads and vibrations.
    """
    M = torch.zeros((num_dof, num_dof))# mass matrix
    m = rho*A*L/420*torch.tensor([[156,   22*L,     54,   -13*L],
                                  [ 22*L,  4*L**2,  13*L, -3*L**2],
                                  [ 54,   13*L,    156,   -22*L],
                                  [-13*L, -3*L**2, -22*L, 4*L**2],])
    for i in range(num_ele):
        M[2*i:2*(i+2), 2*i:2*(i+2)] = M[2*i:2*(i+2), 2*i:2*(i+2)] + m
    
    return M

def F(t, num_ele, L, accel, A, rho):
    ret = torch.tensor([[0], [0]])
    w = rho * accel[t] * A
    for i in range(num_ele):
        if i==num_ele-1:
            ret = torch.cat((ret, torch.tensor([[-w*L/2], [w*L**2/12]])))
        else:
            ret = torch.cat((ret, torch.tensor([[-w*L], [0]])))
    return ret
