import numpy as np
import math
from implicit_solver import thomas_algo_ranna
r   = 0.05
vol = 0.2
ds  = 5
dt  = 0.005
K   = 100

Smin = 0 
Smax = 400
rannacher_step = 2

vec_vn = thomas_algo_ranna(rannacher_step)

tau_n = rannacher_step * dt

# Ai, Bi, Ci coefficients for CN method
def Calc_Ai(vol,r,i):
    return 0.5*(vol**2)*i**2 - 0.5*r*i

def Calc_Bi(vol,r,i):
    return -((vol**2)*(i**2)+r)

def Calc_Ci(vol,r,i):
    return 0.5*(vol**2)*(i**2)+0.5*r*i

def call_boundary(Smax, K, r, tau):
    return Smax - K * math.exp(-r * tau)

## Calculating d vector
# Formula: (I+dt/2*Lh)Vn
def build_d_vec(vec_vn, tau_n):
    # first interior node to the grid index i=1
    i = 1
    b = Calc_Bi(vol, r, i)
    c = Calc_Ci(vol, r, i)
    bR = 1 + (dt/2) * b
    cR = (dt/2) * c
    vec_d = [(bR*vec_vn[0])+(cR*vec_vn[1])] # Works under iff Smin=0
    # interior nodes i = 2 ... (N-1)
    for i in range(len(vec_vn)-2):
        idx = i + 2
        a = Calc_Ai(vol, r, idx)
        b = Calc_Bi(vol, r, idx)
        c = Calc_Ci(vol, r, idx)
        aR = (dt/2) * a
        bR = 1 + (dt/2) * b
        cR = (dt/2) * c
        val = (aR*vec_vn[i])+(bR*vec_vn[i+1])+(cR*vec_vn[i+2])
        vec_d.append(val)
    # final interior node (including right boundary at time level n)
    vn_right = call_boundary(Smax, K, r, tau_n)
    idx = len(vec_vn)
    a = Calc_Ai(vol, r, idx)
    b = Calc_Bi(vol, r, idx)
    c = Calc_Ci(vol, r, idx)
    aR = (dt/2) * a
    bR = 1 + (dt/2) * b
    cR = (dt/2) * c
    vec_d.append((aR*vec_vn[-2])+(bR*vec_vn[-1])+(cR*vn_right))
    return np.array(vec_d, dtype=float)

# Combines the construction M matrix (I-dt/2*Lh) and finds Vn-1 as well
def thomas_algo_cn(times, vec_vn, tau_n):
    # Initial RHS from payoff at maturity
    vec_d = build_d_vec(vec_vn, tau_n)

    M = int(Smax/ds)
    # Accounting for second boundary term at tau_n+dt
    V_boundary_at_tau_n_plus_1 = call_boundary(Smax, K, r, tau_n+dt) # plus 1 represents 1 time step ahead  
    i_last = M - 1
    c_last = Calc_Ci(vol, r, i_last)
    vec_d[-1] += (dt/2) * c_last * V_boundary_at_tau_n_plus_1

    # Interior unknown indices: i = 1..M-1  (matches vec_d length)
    j = list(range(1, M))  # length M-1

    for step in range(times):
        N = len(vec_d)

        # --- Forward sweep ---
        c_dash = np.zeros(N, dtype=float)
        d_dash = np.zeros(N, dtype=float)

        i0 = j[0]
        a = Calc_Ai(vol, r, i0)
        b = Calc_Bi(vol, r, i0)
        c = Calc_Ci(vol, r, i0)
        aL = -(dt/2) * a
        bL = 1 - (dt/2) * b
        cL = -(dt/2) * c

        c_dash[0] = cL / bL
        d_dash[0] = vec_d[0] / bL

        for k in range(1, N):
            idx = j[k]
            a = Calc_Ai(vol, r, idx)
            b = Calc_Bi(vol, r, idx)
            c = Calc_Ci(vol, r, idx)
            aL = -(dt/2) * a
            bL = 1 - (dt/2) * b
            cL = -(dt/2) * c

            den = bL - (aL * c_dash[k-1])
            c_dash[k] = cL / den
            d_dash[k] = (vec_d[k] - (aL * d_dash[k-1])) / den
        
        # --- Back substitution ---
        x = np.zeros(N, dtype=float)
        x[-1] = d_dash[-1]
        for k in range(N-2, -1, -1):
            x[k] = d_dash[k] - c_dash[k] * x[k+1]

        # If another step remains, rebuild RHS for next solve:
        # vec_d := V^n (current known layer) then apply boundary correction to last entry
        if step != times - 1:
            vec_vn = x.copy()
            tau_n += dt # next step uses tau = 3dt, 4dt, ... as rannacher has ran two loops
            vec_d = build_d_vec(vec_vn, tau_n)
            
            # Accounting for boundary at tau_n+dt
            V_boundary_at_tau_n_plus_1 = call_boundary(Smax, K, r, tau_n+dt) # plus 1 represents 1 time step ahead  
            i_last = M - 1
            c_last = Calc_Ci(vol, r, i_last)
            vec_d[-1] += (dt/2) * c_last * V_boundary_at_tau_n_plus_1
        else:
            vec_out = x.copy()
    return vec_out

vec_out = thomas_algo_cn(10, vec_vn, tau_n)
print(vec_out)