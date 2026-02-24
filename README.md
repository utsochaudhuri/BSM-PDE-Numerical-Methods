# BSM PDE Numerical Methods — Convergence Analysis

This project implements and analyses finite difference (FD) schemes for pricing European call options under the Black-Scholes-Merton (BSM) framework. The focus is on demonstrating correct numerical implementation and verifying convergence behaviour through max error analysis against the analytical BSM solution, across both temporal and spatial refinement.

The methods covered are the **fully implicit scheme**, **Crank-Nicolson (CN)**, and **Crank-Nicolson with Rannacher time-stepping**, each tested with three payoff initialisation approaches: no smoothing, cell average smoothing, and quadratic smoothing — giving nine method-configuration combinations in total.

---

## Table of Contents

- [Background](#background)
- [Methods Implemented](#methods-implemented)
- [Payoff Smoothing Techniques](#payoff-smoothing-techniques)
- [Convergence Studies](#convergence-studies)
- [File Structure](#file-structure)
- [Parameters](#parameters)
- [Results Summary](#results-summary)

---

## Background

The BSM PDE for a European call option is:

$$\frac{\partial V}{\partial \tau} = \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV$$

where $\tau = T - t$ is the time to maturity, $S$ is the stock price, $\sigma$ is the volatility, and $r$ is the risk-free rate. This is solved forward in $\tau$ (backward from expiry) on a uniform spatial grid over $S \in [0, S_{\max}]$.

**Boundary conditions:**
- $V(0, \tau) = 0$ (call is worthless when $S = 0$)
- $V(S_{\max}, \tau) = S_{\max} - Ke^{-r\tau}$ (deep in-the-money limit)

**Initial condition (payoff at expiry):**

$$V(S, 0) = \max(S - K,\ 0)$$

The analytical BSM price is used as the reference solution to compute the max absolute error across all interior grid nodes at $\tau = T$.

---

## Methods Implemented

### 1. Fully Implicit Scheme (`implicit_solver.py`)

The implicit scheme discretises the PDE so that spatial derivatives are evaluated at the new time level $\tau^{n+1}$. This gives a tridiagonal system at each time step:

$$A_i V_{i-1}^{n+1} + B_i V_i^{n+1} + C_i V_{i+1}^{n+1} = V_i^n$$

with coefficients:

$$A_i = -\frac{dt}{2}\left(\sigma^2 i^2 - ri\right), \quad B_i = 1 + dt\left(\sigma^2 i^2 + r\right), \quad C_i = -\frac{dt}{2}\left(\sigma^2 i^2 + ri\right)$$

This system is solved efficiently using the **Thomas algorithm** (tridiagonal matrix algorithm). The implicit scheme is unconditionally stable but only **first-order accurate in time**: $O(dt)$.

---

### 2. Crank-Nicolson Scheme (`cn_solver.py`)

The CN scheme averages the spatial operator between the current and next time levels:

$$\left(I - \frac{dt}{2}L_h\right)V^{n+1} = \left(I + \frac{dt}{2}L_h\right)V^n$$

where $L_h$ is the discrete spatial operator. This produces a tridiagonal system that is also solved via the Thomas algorithm. CN is **second-order accurate in both time and space**: $O(dt^2, ds^2)$, but it is sensitive to non-smooth initial conditions — the kink in the call payoff at the strike $K$ can cause spurious oscillations near the strike, particularly at coarser time steps.

---

### 3. Crank-Nicolson with Rannacher Time-Stepping (`implicit_solver.py` + `cn_solver.py`)

The Rannacher scheme addresses the oscillation issue by replacing the first few CN steps with fully implicit steps. Here, 2 implicit half-steps (effectively one full time step) are taken using $dt/2$ before switching to standard CN for the remainder. This suppresses the high-frequency error components introduced by the payoff discontinuity, restoring clean **second-order convergence** without sacrificing long-run accuracy.

---

## Payoff Smoothing Techniques

All three smoothing functions are defined in `payoff.py` and applied at the initial time level ($\tau = 0$).

### No Smoothing
The raw payoff $\max(S - K, 0)$ is used directly. The discontinuity in the first derivative at $S = K$ is unmodified.

### Cell Average Smoothing
Each grid value is replaced by the average of the payoff over the surrounding cell $[S_i - ds/2,\ S_i + ds/2]$:

$$V_i = \frac{1}{ds} \int_{S_i - ds/2}^{S_i + ds/2} \max(s - K,\ 0)\ ds$$

Evaluating this analytically gives:

$$V_i = \begin{cases} 0 & \text{if } S_i + ds/2 \leq K \\ S_i - K & \text{if } S_i - ds/2 \geq K \\ \dfrac{(S_i + ds/2 - K)^2}{2\,ds} & \text{if } S_i - ds/2 < K < S_i + ds/2 \end{cases}$$

This smooths the kink by distributing the discontinuity across one grid cell and is the most effective of the three techniques.

### Quadratic Smoothing
A local quadratic polynomial $p(x) = ax^2 + bx + c$ is fitted over the transition region $x \in [K - ds,\ K + ds]$ (where $x = S - K$), matching the value and derivative of the payoff outside this region:

$$V_i = \begin{cases} 0 & \text{if } S_i < K - ds \\ \dfrac{(S_i - K)^2}{4\,ds} + \dfrac{S_i - K}{2} + \dfrac{ds}{4} & \text{if } K - ds \leq S_i \leq K + ds \\ S_i - K & \text{if } S_i > K + ds \end{cases}$$

This provides a $C^1$-smooth approximation to the payoff over a two-cell transition band.

---

## Convergence Studies

### Temporal Convergence (`error_convg_plot_dt.py`)

The spatial step is fixed at $ds = 0.625$ and the time step is varied as:

$$dt \in \{0.04,\ 0.02,\ 0.01,\ 0.005,\ 0.0025\}$$

The max absolute error against the BSM solution is plotted on a log-log scale for all nine method-configuration combinations. On a log-log plot, a slope of $p$ corresponds to $p$-th order convergence in time.

### Spatial Convergence (`error_convg_plot_ds.py`)

To isolate spatial error, the time step is coupled to the spatial step via $dt \propto ds^2$ (specifically $dt = 0.0002 \cdot ds^2$) to ensure temporal errors remain subdominant. The spatial step is varied as:

$$ds \in \{10,\ 5,\ 2.5,\ 1.25,\ 0.625\}$$

A slope of 2 on the log-log plot confirms second-order spatial convergence.

---

## File Structure

```
.
├── payoff.py                  # Payoff smoothing functions (cell average, quadratic)
├── implicit_solver.py         # Fully implicit Thomas algorithm solver
├── cn_solver.py               # Crank-Nicolson Thomas algorithm solver
├── error_convg_plot_dt.py     # Temporal convergence plots (fixed ds, varying dt)
└── error_convg_plot_ds.py     # Spatial convergence plots (varying ds, dt ∝ ds²)
```

---

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $r$ | 0.05 | Risk-free interest rate |
| $\sigma$ | 0.2 | Volatility |
| $K$ | 100 | Strike price |
| $S_{\max}$ | 400 | Upper boundary of stock price grid |
| $T$ | 1 year | Time to maturity |
| Rannacher steps | 2 | Number of implicit half-steps before switching to CN |

---

## Raw Output

The print blocks in `error_convg_plot_dt.py` and `error_convg_plot_ds.py` are commented out by default. Uncommenting them will print a formatted table of max errors for all nine method-configuration combinations at each refinement level, which can be useful for inspecting the raw values directly.

---

## Results Summary

**Temporal convergence (fixed ds = 0.625):**
- The implicit scheme confirms first-order convergence (slope ≈ 1) on the log-log plot, consistent with $O(dt)$ theory.
- CN without Rannacher exhibits erratic convergence at coarser time steps due to the payoff discontinuity, before settling near the same error floor as CN + Rannacher.
- CN + Rannacher converges smoothly and monotonically from the outset, recovering stable second-order behaviour.
- All methods plateau around $10^{-3}$ at fine time steps, indicating spatial error becomes the bottleneck at $ds = 0.625$.

**Payoff smoothing effect on CN (no Rannacher, temporal study):**
- Cell average smoothing produces the lowest errors across all time steps, converging to a floor roughly half an order of magnitude below the other two methods.
- Quadratic smoothing offers a modest improvement over no smoothing at coarser time steps, with both methods converging to similar errors at finer resolutions.

**Spatial convergence (dt ∝ ds²):**
- No smoothing and quadratic smoothing track nearly identically, both showing clean second-order spatial convergence (slope ≈ 2).
- Cell average smoothing maintains the same second-order convergence rate but with a consistently lower error prefactor — roughly one order of magnitude better across all grid refinements.
- Adding Rannacher time-stepping produces no meaningful difference in the spatial convergence study, confirming that its benefit is purely temporal.
