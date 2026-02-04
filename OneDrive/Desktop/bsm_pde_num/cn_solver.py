import numpy as np
import math
from implicit_solver import thomas_algo_ranna
r   = 0.05
vol = 0.2
ds  = 10
dt  = 0.005
K   = 100

Smin = 0 
Smax = 400
rannacher_step = 2

vec_vn = thomas_algo_ranna(rannacher_step)

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
def build_d_1st_step():
    # first interior node to the grid index i=1
    i = 1
    a = Calc_Ai(vol, r, i)
    b = Calc_Bi(vol, r, i)
    c = Calc_Ci(vol, r, i)
    aR = (dt/2) * a
    bR = 1 + (dt/2) * b
    cR = (dt/2) * c
    vec_d = [(bR*vec_vn[0])+(cR*vec_vn[1])]
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
    tau_n = rannacher_step * dt
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
def thomas_algo_cn(times):
    # Initial RHS from payoff at maturity
    vec_d = build_d_1st_step()  

    M = int(Smax/ds)
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
            tau = (step + 2 + rannacher_step) * dt  # next step uses tau = 2dt, 3dt, ...
            vec_d = x.copy()
            V_boundary = call_boundary(Smax, K, r, tau)
            i_last = M - 1
            c_last = Calc_Ci(vol, r, i_last)
            cL_last = -(dt/2) * c_last
            vec_d[-1] = vec_d[-1] - (cL_last * V_boundary)
        else:
            vec_d = x.copy()
    
    vec_d = [round(val,4) for val in vec_d]
    return vec_d