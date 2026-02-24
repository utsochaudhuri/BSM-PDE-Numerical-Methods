def cell_average_smoothing(ds, K, N):
    smoothed = []
    for idx in range(N):
        Si = (idx+1) * ds
        a = Si-ds/2
        b = Si+ds/2
        # obtained below values/formulae from simplifying 1/ds * integration of max(s-K,0)ds
        if b <= K:
            val = 0
        elif a >= K:
            val = Si-K 
        else: # a<K<b
            val = ((b-K)**2) / (2*ds)
        
        smoothed.append(val)
    
    return smoothed

def quadratic_smoothing(ds, K, N):
    smoothed = []
    for idx in range(N): # note ε=ds
        Si = (idx+1) * ds
        if Si < K-ds:
            val = 0
        elif Si > K+ds:
            val = Si-K
        else: # K-ds<=Si<=K+ds
            # derived from qudratic fitting of p(x) = ax^2+bx+c over x ∈ [K-ds, K+ds]
            # where value and derivative term match outside range conditions
            x = Si-K
            val = (x**2)/(4*ds) + x/2 + ds/4 

        smoothed.append(val)
    
    return smoothed