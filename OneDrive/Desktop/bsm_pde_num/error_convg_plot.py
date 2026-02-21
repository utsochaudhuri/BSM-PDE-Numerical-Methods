from implicit_solver import thomas_algo_ranna
from scipy.stats import norm
import math
from cn_solver import thomas_algo_cn

# Black Scholes for given call option price
def bsm_call_price(S):
    if S == 0:
        return 0.0
    
    d1 = (math.log(S/K) + (r + 0.5*vol**2)*T) / (vol * math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)

    return S*norm.cdf(d1)-K*(math.e**(-r*T))*norm.cdf(d2)


dt_vals = [0.04, 0.02, 0.01, 0.005, 0.0025]

for dt in dt_vals:
    r=0.05
    vol=0.2
    ds = 0.625
    K=100

    Smin = 0
    Smax = 400
    T = 1 # Time to maturity (In years)
    loop_steps = int(T/dt)

    vec_vn_implicit = thomas_algo_ranna(loop_steps, ds, dt)

    errors = []
    for idx in range(len(vec_vn_implicit)):
        bsm_price = bsm_call_price((idx+1)*ds)
        error = abs(vec_vn_implicit[idx] - bsm_price)
        errors.append(error)

    max_error = max(errors)
    print("implicit")
    print("dt_value", dt, ": ", max_error)

    