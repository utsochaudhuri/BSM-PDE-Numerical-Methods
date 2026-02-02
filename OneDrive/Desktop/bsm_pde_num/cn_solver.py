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

vec_vn = thomas_algo_ranna(2)

def Calc_Ai(vol,r,i):
    return 0.5*(vol**2)*i**2 - 0.5*r*i

def Calc_Bi(vol,r,i):
    return -((vol**2)*(i**2)+r)

def Calc_Ci(vol,r,i):
    return 0.5*(vol**2)*(i**2)+0.5*r*i

## Calculating d vector
# Formula: (I+dt/2*Lh)Vn

def build_d_1st_step():
    vec_d = [(1+dt/2*Calc_Bi(vol,r,1)*vec_vn[0])+(dt/2*Calc_Ci(vol,r,1)*vec_vn[1])]
    for i in range(len(vec_vn)-2):
        val = (dt/2*Calc_Ai(vol,r,i+2)*vec_vn[i])+(1+dt/2*Calc_Bi(vol,r,i+2)*vec_vn[i+1])+(dt/2*Calc_Ci(vol,r,i+2)*vec_vn[i+2])
        vec_d.append(val)
    vec_d.append((dt/2*Calc_Ai(vol,r,len(vec_vn))*vec_vn[-2])+(1+dt/2*Calc_Bi(vol,r,len(vec_vn))*vec_vn[-1]))
    vec_d = np.array(vec_d, dtype=float)
    return vec_d

print(build_d_1st_step())