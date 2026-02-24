from implicit_solver import thomas_algo_ranna, payoff_call
from scipy.stats import norm
import math
from cn_solver import thomas_algo_cn
from payoff import cell_average_smoothing, quadratic_smoothing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Black Scholes for given call option price
def bsm_call_price(S):
    if S == 0:
        return 0.0
    
    d1 = (math.log(S/K) + (r + 0.5*vol**2)*T) / (vol * math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)

    return S*norm.cdf(d1)-K*(math.e**(-r*T))*norm.cdf(d2)

r=0.05
vol=0.2
K=100
Smin = 0
Smax = 400
T = 1 # Time to maturity (In years)
ds_vals = [10, 5, 2.5, 1.25, 0.625] # Values of ds for plot (in form ds,ds/2,ds/4,...)

# Array intialisation for error values
max_err_imp_arr = []
max_err_imp_cell_arr = []
max_err_imp_quad_arr = []
max_err_cn_arr = []
max_err_cn_cell_arr = []
max_err_cn_quad_arr = []
max_err_cn_ran_arr = []
max_err_cn_ran_cell_arr = []
max_err_cn_ran_quad_arr = []

for ds in ds_vals:
    # Ensures loop ends at T exactly
    dt = 0.0002 * (ds**2)
    loop_steps = int(round(T/dt))
    dt = T/loop_steps

    vec_vn_implicit = thomas_algo_ranna(loop_steps, ds, dt, 0)
    vec_vn_implicit_cell_averaged = thomas_algo_ranna(loop_steps, ds, dt, 1)
    vec_vn_implicit_quadratic_smooth = thomas_algo_ranna(loop_steps, ds, dt, 2)

    errors_implicit = []
    errors_implicit_cell_averaged = []
    errors_implicit_quadratic_smooth = []

    vec_vn_cn = []
    vec_vn_cn_cell_averaged = cell_average_smoothing(ds, K, len(vec_vn_implicit))
    vec_vn_cn_quadratic_smooth = quadratic_smoothing(ds, K, len(vec_vn_implicit))

    for idx in range(len(vec_vn_implicit)):
        bsm_price = bsm_call_price((idx+1)*ds)
        # Implicit method with no smoothening
        error_implicit = abs(vec_vn_implicit[idx] - bsm_price)
        errors_implicit.append(error_implicit)
        # Implicit method with cell average smoothening
        error_implicit_cell_averaged = abs(vec_vn_implicit_cell_averaged[idx] - bsm_price)
        errors_implicit_cell_averaged.append(error_implicit_cell_averaged)
        # Implicit method with quadratic smoothening
        error_implicit_quadratic_smooth = abs(vec_vn_implicit_quadratic_smooth[idx] - bsm_price)
        errors_implicit_quadratic_smooth.append(error_implicit_quadratic_smooth)

        vec_vn_cn.append(payoff_call((idx+1)*ds)) # Simulatenous vec_vn contrsuction for cn method w/o rannacher

    ## CN method without rannacher (smooth, cell averaged, quadratic smooth)
    vec_vn_cn_out = thomas_algo_cn(loop_steps, vec_vn_cn, 0, ds, dt) # note tau_n is 0 as it's w/o rannacher steps
    vec_vc_cn_out_cell_averaged = thomas_algo_cn(loop_steps, vec_vn_cn_cell_averaged, 0, ds, dt)
    vec_vc_cn_out_quadratic_smooth = thomas_algo_cn(loop_steps, vec_vn_cn_quadratic_smooth, 0, ds, dt)

    ## CN method with rannacher
    rannacher_step = 2
    tau_n = rannacher_step * dt/2
    # No smoothening
    vec_vn_rannach = thomas_algo_ranna(rannacher_step, ds, dt/2, 0)
    vec_vc_cn_ran_out = thomas_algo_cn(loop_steps-1, vec_vn_rannach, tau_n, ds, dt) # loop-1 accounting
    # Cell average smoothening
    vec_vn_rannach_cell_averaged = thomas_algo_ranna(rannacher_step, ds, dt/2, 1)
    vec_vc_cn_ran_out_cell_averaged = thomas_algo_cn(loop_steps-1, vec_vn_rannach_cell_averaged, tau_n, ds, dt)
    # Quadratic smoothening
    vec_vn_rannach_quadratic_smooth = thomas_algo_ranna(rannacher_step, ds, dt/2, 2)
    vec_vc_cn_ran_out_quadratic_smooth = thomas_algo_cn(loop_steps-1, vec_vn_rannach_quadratic_smooth, tau_n, ds, dt)

    # Error calculations for CN method (w and w/o rannacher + smoothening techniques)
    errors_cn_wo_ran = []
    errors_cn_wo_ran_cell_averaged = []
    errors_cn_wo_ran_quadratic_smooth = []

    errors_cn_ran = []
    errors_cn_ran_cell_averaged = []
    errors_cn_ran_quadratic_smooth = []

    # Error calculation for cn method (w and w/o rannacher)
    for idx in range(len(vec_vn_implicit)):
        bsm_price = bsm_call_price((idx+1)*ds)

        # error w/o rannacher (all smoothening techniques)
        error_cn_wo_ran = abs(vec_vn_cn_out[idx] - bsm_price) 
        errors_cn_wo_ran.append(error_cn_wo_ran)
        error_cn_wo_ran_cell_averaged = abs(vec_vc_cn_out_cell_averaged[idx] - bsm_price) 
        errors_cn_wo_ran_cell_averaged.append(error_cn_wo_ran_cell_averaged)
        error_cn_wo_ran_quadratic_smooth = abs(vec_vc_cn_out_quadratic_smooth[idx] - bsm_price)
        errors_cn_wo_ran_quadratic_smooth.append(error_cn_wo_ran_quadratic_smooth)

        # error w rannacher (all smoothening techniques)
        error_cn_ran = abs(vec_vc_cn_ran_out[idx] - bsm_price) 
        errors_cn_ran.append(error_cn_ran)
        error_cn_ran_cell_averaged = abs(vec_vc_cn_ran_out_cell_averaged[idx] - bsm_price) 
        errors_cn_ran_cell_averaged.append(error_cn_ran_cell_averaged)
        error_cn_ran_quadratic_smooth = abs(vec_vc_cn_ran_out_quadratic_smooth[idx] - bsm_price) 
        errors_cn_ran_quadratic_smooth.append(error_cn_ran_quadratic_smooth)

    max_err_imp = max(errors_implicit)
    max_err_imp_arr.append(max(errors_implicit))
    max_err_imp_cell = max(errors_implicit_cell_averaged)
    max_err_imp_cell_arr.append(max(errors_implicit_cell_averaged))
    max_err_imp_quad = max(errors_implicit_quadratic_smooth)
    max_err_imp_quad_arr.append(max(errors_implicit_quadratic_smooth))

    max_err_cn = max(errors_cn_wo_ran)
    max_err_cn_arr.append(max_err_cn)
    max_err_cn_cell = max(errors_cn_wo_ran_cell_averaged)
    max_err_cn_cell_arr.append(max_err_cn_cell)
    max_err_cn_quad = max(errors_cn_wo_ran_quadratic_smooth)
    max_err_cn_quad_arr.append(max_err_cn_quad)

    max_err_cn_ran = max(errors_cn_ran)
    max_err_cn_ran_arr.append(max_err_cn_ran)
    max_err_cn_ran_cell = max(errors_cn_ran_cell_averaged)
    max_err_cn_ran_cell_arr.append(max_err_cn_ran_cell)
    max_err_cn_ran_quad = max(errors_cn_ran_quadratic_smooth)
    max_err_cn_ran_quad_arr.append(max_err_cn_ran_quad)

    # # Uncomment to get raw values (Picked up from chatgpt)
    # print("="*70)
    # print(f"dt = {dt:.6f} | ds = {ds}")
    # print("-"*70)

    # print("IMPLICIT METHOD")
    # print(f"  Raw payoff               : {max_err_imp:.10e}")
    # print(f"  Cell-average smoothing   : {max_err_imp_cell:.10e}")
    # print(f"  Quadratic smoothing      : {max_err_imp_quad:.10e}")

    # print("\nCRANK-NICOLSON (no Rannacher)")
    # print(f"  Raw payoff               : {max_err_cn:.10e}")
    # print(f"  Cell-average smoothing   : {max_err_cn_cell:.10e}")
    # print(f"  Quadratic smoothing      : {max_err_cn_quad:.10e}")

    # print("\nCRANK-NICOLSON (with Rannacher)")
    # print(f"  Raw payoff               : {max_err_cn_ran:.10e}")
    # print(f"  Cell-average smoothing   : {max_err_cn_ran_cell:.10e}")
    # print(f"  Quadratic smoothing      : {max_err_cn_ran_quad:.10e}")

    # print("="*70)
    # print()


## Plot 1 (spatial convergence of cn method without rannacher)
plt.figure(figsize=(10, 5))

plt.loglog(ds_vals, max_err_cn_arr, 'o-', label='No smoothening')
plt.loglog(ds_vals, max_err_cn_cell_arr, 's-', label='Cell average smoothening')
plt.loglog(ds_vals, max_err_cn_quad_arr, '^-', label='Quadratic smoothening')

plt.xlabel("Spatial step (ds)")
plt.ylabel("Max error")
plt.title(f"Spatial Convergence of Crank Nicolson with Payoff smoothing (no Rannacher, log-log, dt∝ds², 1 year maturity)")

# --- Force x-axis ticks to be exactly your dt values ---
plt.xticks(ds_vals, [f"{ds:g}" for ds in ds_vals])

# Optional: show refinement left → right (common in convergence plots)
plt.gca().invert_xaxis()

# --- Improve y-axis log tick density ---
plt.gca().yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=10))
plt.gca().yaxis.set_minor_locator(
    mticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=100)
)
plt.gca().yaxis.set_minor_formatter(mticker.NullFormatter())

# Better grid styling
plt.grid(True, which="major", linestyle="--", linewidth=0.8)
plt.grid(True, which="minor", linestyle=":", linewidth=0.5)

plt.legend()
plt.tight_layout()
plt.show()



## Plot 2 (spatial convergence on cn method with rannacher)
plt.figure(figsize=(10, 5))

plt.loglog(ds_vals, max_err_cn_ran_arr, 'o-', label='No smoothening')
plt.loglog(ds_vals, max_err_cn_ran_cell_arr, 's-', label='Cell average smoothening')
plt.loglog(ds_vals, max_err_cn_ran_quad_arr, '^-', label='Quadratic smoothening')

plt.xlabel("Spatial step (ds)")
plt.ylabel("Max error")
plt.title(f"Spatial Convergence of Crank Nicolson with Payoff smoothing (with Rannacher, log-log, dt∝ds², 1 year maturity)")

# --- Force x-axis ticks to be exactly your dt values ---
plt.xticks(ds_vals, [f"{ds:g}" for ds in ds_vals])

# Optional: show refinement left → right (common in convergence plots)
plt.gca().invert_xaxis()

# --- Improve y-axis log tick density ---
plt.gca().yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=10))
plt.gca().yaxis.set_minor_locator(
    mticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=100)
)
plt.gca().yaxis.set_minor_formatter(mticker.NullFormatter())

# Better grid styling
plt.grid(True, which="major", linestyle="--", linewidth=0.8)
plt.grid(True, which="minor", linestyle=":", linewidth=0.5)

plt.legend()
plt.tight_layout()
plt.show()