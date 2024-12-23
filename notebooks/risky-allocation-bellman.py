import numpy as np

def utility(consumption, gamma=2.0, eps=1e-8):
    if consumption < eps:
        return -1e9
    if abs(gamma - 1.0) < 1e-8:
        return np.log(consumption)
    else:
        return consumption**(1-gamma) / (1-gamma)

def solve_consumption_investment(
    W0=1_000_000,
    r_tp=0.02,      # discount rate
    gamma=2.0,      # CRRA param
    T=35,
    n_grid=200,     # wealth grid size
    c_grid_size=21, # discrete c
    k_grid_size=21, # discrete k
    # Risky return distribution (discrete approx):
    R_r_vals = [ -0.20, 0.00, 0.10, 0.20 ],   # possible realized returns
    R_r_probs = [ 0.10,  0.40, 0.40, 0.10 ],  # sum to 1
    r_f = 0.03,      # risk-free rate
):

    # 1) Build wealth grid
    W_max = W0 * (1+r_f)**T * 2.0
    W_grid = np.linspace(1e-3, W_max, n_grid)

    # 2) Storage for value & policy (T+2 to hold V[T+1])
    V = np.zeros((T+2, n_grid))
    C_opt = np.zeros((T+2, n_grid))
    K_opt = np.zeros((T+2, n_grid))

    # 3) Terminal condition: V[T+1] = 0
    #    (no bequest motive)

    # 4) Precompute discount factors
    discount_factors = [(1+r_tp)**t for t in range(T+2)]

    # Discretize c and k
    c_candidates = np.linspace(0, 1, c_grid_size)
    k_candidates = np.linspace(0, 1, k_grid_size)

    # Function to do nearest or linear interpolation of V[t+1](W_next)
    def interpolate_value(W_val, Vrow, Wgrid):
        """Return the interpolated value at W_val in Vrow."""
        if W_val <= Wgrid[0]:
            return Vrow[0]
        if W_val >= Wgrid[-1]:
            return Vrow[-1]
        idx = np.searchsorted(Wgrid, W_val)
        # If exact match
        if W_val == Wgrid[idx]:
            return Vrow[idx]
        # else linear interpolation between idx-1 and idx
        w1 = Wgrid[idx-1]
        w2 = Wgrid[idx]
        v1 = Vrow[idx-1]
        v2 = Vrow[idx]
        return v1 + (v2 - v1)*(W_val - w1)/(w2 - w1)

    # 5) Backward induction
    for t in reversed(range(1, T+1)):
        df = discount_factors[t]
        for i, W_now in enumerate(W_grid):
            
            best_val = -np.inf
            best_c = 0
            best_k = 0
            
            for c_frac in c_candidates:
                cons = c_frac * W_now
                # If we want to skip c_frac close to 1 (or 0) ...
                
                for k_frac in k_candidates:
                    # Next wealth depends on the random return scenario:
                    exp_future_val = 0.0
                    for Rr, prob in zip(R_r_vals, R_r_probs):
                        # Realized portfolio return
                        Rp = k_frac*Rr + (1 - k_frac)*r_f
                        W_next = (W_now - cons)*(1 + Rp)
                        
                        # Interpolate V[t+1]
                        v_next = interpolate_value(W_next, V[t+1], W_grid)
                        exp_future_val += prob * v_next

                    # immediate utility
                    u = utility(cons, gamma=gamma)
                    discounted_u = u / df

                    total_val = discounted_u + exp_future_val

                    if total_val > best_val:
                        best_val = total_val
                        best_c = c_frac
                        best_k = k_frac

            V[t, i] = best_val
            C_opt[t, i] = best_c
            K_opt[t, i] = best_k

    # 6) Extract optimal actions for the initial wealth W0 at t=1
    i_closest = np.argmin(np.abs(W_grid - W0))
    c_init = C_opt[1, i_closest]
    k_init = K_opt[1, i_closest]

    return V, C_opt, K_opt, W_grid, (c_init, k_init)

def main():
    V, C_opt, K_opt, W_grid, (c_init, k_init) = solve_consumption_investment()
    print(f"Optimal initial consumption fraction: {c_init:.3f}")
    print(f"Optimal initial fraction in risky asset: {k_init:.3f}")

if __name__ == "__main__":
    main()
