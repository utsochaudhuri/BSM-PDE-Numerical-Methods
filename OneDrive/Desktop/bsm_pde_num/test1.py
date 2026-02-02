import numpy as np
import math

r=0.05
vol=0.2
ds=10
dt=0.005
K=100

def Calc_Ai(dt,vol,r,i):
    return -0.5*dt*((vol**2)*(i**2)-r*i)

def Calc_Bi(dt,vol,r,i):
    return 1+dt*((vol**2)*(i**2)+r)

def Calc_Ci(dt,vol,r,i):
    return -0.5*dt*((vol**2)*(i**2)+r*i)

Smin = 0
Smax = 400

def payoff_call(S):
    return max(S-K, 0.0)

def call_boundary(Smax, K, r, tau):
    return Smax - K * math.exp(-r * tau)

def build_b_1st_step():
    tau = dt
    vec_b = []
    for S in range(Smin+ds, Smax, ds):  # interior nodes only: 10..390
        vec_b.append(payoff_call(S))
    # boundary correction applied to LAST interior node (Smax-ds -> i = M-1)
    V_boundary = call_boundary(Smax, K, r, tau)
    M = int(Smax/ds)
    i_last = M - 1
    vec_b[-1] = vec_b[-1] - Calc_Ci(dt, vol, r, i_last) * V_boundary
    return np.array(vec_b, dtype=float)

def thomas_algo(times):
    # Initial RHS from payoff at maturity
    vec_b = build_b_1st_step()

    M = int(Smax/ds)
    # Interior unknown indices: i = 1..M-1  (matches vec_b length)
    j = list(range(1, M))  # length M-1

    for step in range(times):
        N = len(vec_b)

        # --- Forward sweep ---
        c_dash = np.zeros(N, dtype=float)
        d_dash = np.zeros(N, dtype=float)

        c_dash[0] = Calc_Ci(dt, vol, r, j[0]) / Calc_Bi(dt, vol, r, j[0])
        d_dash[0] = vec_b[0] / Calc_Bi(dt, vol, r, j[0])

        for k in range(1, N):
            idx = j[k]
            den = Calc_Bi(dt, vol, r, idx) - Calc_Ai(dt, vol, r, idx) * c_dash[k-1]
            c_dash[k] = Calc_Ci(dt, vol, r, idx) / den
            d_dash[k] = (vec_b[k] - Calc_Ai(dt, vol, r, idx) * d_dash[k-1]) / den

        # --- Back substitution ---
        x = np.zeros(N, dtype=float)
        x[-1] = d_dash[-1]
        for k in range(N-2, -1, -1):
            x[k] = d_dash[k] - c_dash[k] * x[k+1]

        # If another step remains, rebuild RHS for next solve:
        # vec_b := V^n (current known layer) then apply boundary correction to last entry
        if step != times - 1:
            tau = (step + 2) * dt  # next step uses tau = 2dt, 3dt, ...
            vec_b = x.copy()
            V_boundary = call_boundary(Smax, K, r, tau)
            i_last = M - 1
            vec_b[-1] = vec_b[-1] - Calc_Ci(dt, vol, r, i_last) * V_boundary
        else:
            vec_b = x.copy()

    vec_b = [round(val,4) for val in vec_b]
    return vec_b

output = thomas_algo(2)
a = [num for num in range(10,410,10)]
res = dict(zip(a,output))
print(res)