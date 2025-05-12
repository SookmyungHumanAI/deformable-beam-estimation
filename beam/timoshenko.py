import torch
from math import pi


def assemble_K_timo(num_ele, num_dof, E, I, L, d, nu = 0.35):
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

def assemble_M_timo_(num_ele, num_dof, rho, A, L):
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
def assemble_M_timo(num_ele, num_dof, rho, A, L, I):
    """
    Assembles the global mass matrix for a Timoshenko beam element system (2D, 2-node element).

    Parameters
    ----------
    num_ele : int
        요소(Elements) 개수
    num_dof : int
        전체 자유도(= 2*(num_ele+1))
    rho : float
        재료 밀도 [kg/m^3]
    A : float
        (원형) 단면적
    L : float
        요소 길이
    I : float
        단면 2차 모멘트(회전 관성 계산 시 사용)

    Returns
    -------
    M : (num_dof x num_dof) torch.Tensor
        전역(global) 질량 행렬
    """

    M = torch.zeros((num_dof, num_dof), dtype=torch.float)

    # -----------------------
    # (1) Translational mass (일관질량 행렬, Euler-Bernoulli 부분)
    # -----------------------
    #   m_t = (rho * A * L / 420) * [...]
    m_t = rho * A * L / 420.0 * torch.tensor([
        [156.0,     22.0*L,     54.0,       -13.0*L  ],
        [22.0*L,    4.0*L**2,   13.0*L,     -3.0*L**2],
        [54.0,      13.0*L,     156.0,      -22.0*L  ],
        [-13.0*L,   -3.0*L**2,  -22.0*L,    4.0*L**2 ]
    ], dtype=torch.float)

    # -----------------------
    # (2) Rotary inertia term (회전 관성)
    # -----------------------
    #   m_r = ( rho * I / (420 * L ) ) * [...]
    #   (Timoshenko에서는 단면 높이가 큰 경우 회전관성이 더 크게 작용)
    m_r = rho * I / (420.0 * L) * torch.tensor([
        [36.0,    3.0*L,    -36.0,    3.0*L],
        [3.0*L,   4.0*L**2, -3.0*L,  -1.0*L**2],
        [-36.0,   -3.0*L,    36.0,   -3.0*L],
        [3.0*L,  -1.0*L**2, -3.0*L,   4.0*L**2]
    ], dtype=torch.float)

    # 두 부분을 합산 -> 해당 요소의 (4x4) Timoshenko 질량 행렬
    m_local = m_t + m_r

    # 전역조립
    for i in range(num_ele):
        idx = slice(2*i, 2*i+4)
        M[idx, idx] += m_local

    return M

# def init_disp(num_dof, M, K, end_time, dt=0.0001, F0=None):
#     """
#     Initializes displacement, velocity, and acceleration arrays for dynamic analysis.

#     Parameters:
#     - M (torch.Tensor): The global mass matrix of the structural system
#     - K (torch.Tensor): The global stiffness matrix of the structural system
#     - dt (float): Time step for the analysis
#     - F0 (torch.Tensor): Initial external force vector applied to the system
#     - num_dof
#     - end_time

#     Returns:
#     - disp (torch.Tensor): A tensor containing the initialized displacements for each degree of freedom (DoF) at each time step.

#     This function initializes displacement, velocity, and acceleration arrays for dynamic structural analysis.
#     It calculates initial accelerations based on the provided initial force `F0` and the mass (`M`) and stiffness (`K`) matrices,
#     which are assumed to be defined outside this function. 

#     To solve the dynamic equations using the central difference method,
#     the function computes the initial displacement `d1` for the first time step.
#     It assumes zero initial displacements (`d0`) and velocities (`v0`). 
#     """

#     delta_t = torch.tensor([dt])
#     num_steps = int(end_time/dt)
    
#     disp = torch.zeros((num_dof, num_steps + 1)) # (δ, θ)_1, (δ, θ)_2, ..., (δ, θ)_(num_node)
#     velocities = torch.zeros((num_dof, num_steps + 1))
#     accelerations = torch.zeros((num_dof, num_steps + 1))
    
#     d0 = torch.zeros((num_dof, 1))
#     v0 = torch.zeros((num_dof, 1))
#     a0 = torch.zeros((num_dof, 1))
    
#     disp[:, 0] = d0.view(-1)
#     velocities[:, 0] = v0.view(-1)
#     accelerations[:, 0] = a0.view(-1)
    
#     if F0==None:
#         raise ValueError("Enter the initial value of External Force")
    
#     accelerations[2:, 0] = torch.linalg.inv(M[2:, 2:]) @ (F0 - K[2:, 2:] @ d0[2:]).view(-1)
#     d_minus_1 = d0[2:] - delta_t * v0[2:] + (delta_t**2) / 2 * accelerations[2:, [0]] # d_(-1) initial value
#     d1 = torch.linalg.inv(M[2:, 2:]) \
#         @ (delta_t**2*F0 + (2*M[2:, 2:] - (delta_t**2)*K[2:, 2:])@disp[2:,[0]]-M[2:, 2:]@d_minus_1).view(-1)
#     disp[2:, 1] = d1.view(-1)
    
#     return disp
