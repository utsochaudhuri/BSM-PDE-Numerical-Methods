import numpy as np
import math

r=0.05 # Interest rate
vol=0.2 # Sigma value (volatility)
ds=10 # Delta in stock grid
dt=0.005 # Delta in time (1/number of steps to maturity)
K=100 # Strike price

def Calc_Ai(dt,vol,r,i):
    return -1/2*dt*((vol**2)*(i**2)-r*i)
def Calc_Bi(dt,vol,r,i):
    return 1+dt*((vol**2)*(i**2)+r)
def Calc_Ci(dt,vol,r,i):
    return -1/2*dt*((vol**2)*(i**2)+r*i)

## Implicit method 
i = int(K/ds)
Smin = 0
Smax = 400

# Payoff at maturity
def payoff_call(S):
    return max(S-K, 0.0)

# Call boundary
def call_boundary(Smax, K, r, tau):
    return Smax - K * math.exp(-r * tau)

def build_b_1st_step():
    tau = dt
    vec_b = np.array([])
    for S in range(Smin+ds, Smax, ds):
        if S != Smax-ds:
            vec_b = np.append(vec_b, payoff_call(S))
        else:
            V_boundary = call_boundary(Smax, K, r, tau)
            dn = payoff_call(S) - Calc_Ci(dt,vol,r,S//ds)*V_boundary
            vec_b = np.append(vec_b, dn)
    return vec_b

def thomas_algo(times):
    vec_b = build_b_1st_step()
    j = [S for S in range(int(Smin/ds), int(Smax/ds))]
    for step in range(times):
        c_dash_i = Calc_Ci(dt,vol,r,j[0]) / Calc_Bi(dt,vol,r,j[0])
        d_dash_i = vec_b[0] / Calc_Bi(dt,vol,r,j[0])
        c_dash_i_arr = [c_dash_i]
        d_dash_i_arr = [d_dash_i]
        for I in range(0,len(vec_b)-1):
            den_i = Calc_Bi(dt,vol,r,j[1+I]) - Calc_Ai(dt,vol,r,j[1+I])*c_dash_i
            c_dash_i = Calc_Ci(dt,vol,r,j[1+I]) / den_i
            d_dash_i = (vec_b[I+1]-Calc_Ai(dt,vol,r,j[1+I])*d_dash_i)/den_i
            c_dash_i_arr.append(c_dash_i)
            d_dash_i_arr.append(d_dash_i)
        x_arr = []
        for i in range(int(Smax/ds)-2, int(Smin/ds)-1, -1):
            if i != int(Smax/ds) - 2:
                x_i = d_dash_i_arr[i] - c_dash_i_arr[i]*x_arr[-1]
                x_arr.append(x_i)
            else:
                xn = d_dash_i_arr[i]
                x_arr.append(xn)

        if step != times-1:
            tau = (2 + step) * dt
            x_arr[0] = x_arr[0]-Calc_Ci(dt,vol,r,j[-1])*call_boundary(Smax, K, r, tau)
            x_arr = x_arr[::-1]
            vec_b = np.array(x_arr)
        else:
            x_arr = x_arr[::-1]
            vec_b = np.array(x_arr)
    return vec_b

output = thomas_algo(2)
print(output)