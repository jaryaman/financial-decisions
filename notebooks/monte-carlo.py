import numpy as np

# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
def crra_utility(consumption, gamma=2.0, subsistence=30000, gamma_low=5.0):
    """
    Piecewise CRRA:
    - If consumption < subsistence, treat as higher risk aversion gamma_low.
    - Otherwise gamma as normal (e.g. 2).
    """
    if consumption <= 0:
        return -1e12  # Large negative penalty if consumption is zero or negative
    
    if consumption < subsistence:
        this_gamma = gamma_low
    else:
        this_gamma = gamma
    
    if abs(this_gamma - 1.0) < 1e-9:
        # gamma ~ 1 => log utility
        return np.log(consumption)
    else:
        return consumption**(1.0 - this_gamma) / (1.0 - this_gamma)

def bequest_utility(wealth, b=10, gamma=2.0):
    """
    U(Bequest) =  b * [ 1 - (W/b)^(1 - gamma ) ] / (gamma - 1)
    for example gamma=2, b=10, etc.
    """
    if wealth <= 0:
        return 0.0  # or negative utility, but typically 0 is fine if no wealth
    # If gamma=2 => b * [1 - (W/b)^(-1)] / (1)
    # More general:
    return b * (1 - (wealth / b)**(1 - gamma)) / (gamma - 1)


# -------------------------------------------------------------------
# Stochastic Returns: lognormal for stocks
# -------------------------------------------------------------------
def draw_lognormal_return(mean_return=0.09, stdev=0.20, n_sims=1):
    """
    Draw from a lognormal distribution with approximate mean=mean_return,
    stdev=stdev (both in a typical 'arithmetic' sense).
    
    For a lognormal with underlying normal(m, s), 
    Expected value = exp(m + s^2/2).
    
    We want E(R) ~ mean_return. 
    One approach: 
      let mu_normal = ln(1 + mean_return) - 0.5 * stdev^2
      (assuming stdev^2 is variance in log space)
    Then sample from lognormal with that mu and s = stdev in log space.
    
    But note: If stdev=0.2 (20%), we might interpret that as the 
    stdev of the *arithmetic* returns or log returns. There's some confusion.
    
    For simplicity, let's define:
      lognormal_params = (mu_log, sigma_log)
    such that mean of R ~ mean_return, 
    and sigma of R ~ stdev, roughly.
    
    We'll do a rough approach here. Real calibration is more involved.
    """
    # A simpler approach: draw from normal with mean=mean_return, stdev=stdev,
    # then do (1 + normal_draw).
    # But that can lead to negative returns. Let's do a direct lognormal approach:
    
    # We want to produce returns R >= -100%.
    # Usually we do: R = exp(X) - 1, where X ~ Normal(m, s^2).
    # E[R] = E[exp(X) - 1] = exp(m + s^2/2) - 1.
    
    # So set exp(m + s^2/2) - 1 = mean_return => m + s^2/2 = ln(1 + mean_return).
    
    sigma_log = np.sqrt(np.log(1 + (stdev**2 / (1+mean_return)**2)))  # approximate
    mu_log = np.log(1 + mean_return) - 0.5 * sigma_log**2
    
    # Now draw X ~ Normal(mu_log, sigma_log^2), then R = exp(X)-1
    X = np.random.normal(loc=mu_log, scale=sigma_log, size=n_sims)
    R = np.exp(X) - 1.0
    return R

# -------------------------------------------------------------------
# Policy / Strategy
# -------------------------------------------------------------------
def policy(consum_fraction=0.04, stock_fraction_roth=0.8, stock_fraction_tax=0.6):
    """
    A simple param-based policy: 
    - We withdraw 'consum_fraction' * total_wealth above Social Security 
      each year (plus the guaranteed $30k from Social Security).
    - For Roth: hold 'stock_fraction_roth' in stocks, remainder in safe.
    - For Taxable: hold 'stock_fraction_tax' in stocks, remainder in safe.
    
    Return a dict or tuple with these decisions, or compute them on the fly.
    """
    return {
        "consum_fraction": consum_fraction,
        "stock_fraction_roth": stock_fraction_roth,
        "stock_fraction_tax": stock_fraction_tax
    }


# -------------------------------------------------------------------
# Monte Carlo Simulation
# -------------------------------------------------------------------
def simulate_life_path(
    W_roth_init=400_000,
    W_tax_init=600_000,
    T=35,
    social_security=30_000,
    r_safe=0.04,          # safe asset nominal
    mean_stock=0.09,      # stock arithmetic mean
    stdev_stock=0.20,     # stock volatility
    tax_rate=0.20,        # for taxable account returns
    subsistence=30_000,
    gamma=2.0,
    gamma_low=5.0,        # higher risk aversion below subsistence
    b=10.0,               # bequest param
    r_timepref=0.02,      # discount rate
    survival_prob=0.97,   # each year you have 97% chance to survive
    policy_dict=None,
    rng_seed=None
):
    """
    Simulate one life path (1 scenario) up to T years or death.
    Returns: discounted utility total, final wealth (for debugging), alive_or_not, ...
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)
    
    # If no policy supplied, use a default
    if policy_dict is None:
        policy_dict = policy()
    
    W_roth = W_roth_init
    W_tax = W_tax_init
    
    consum_fraction = policy_dict["consum_fraction"]
    stock_frac_roth = policy_dict["stock_fraction_roth"]
    stock_frac_tax  = policy_dict["stock_fraction_tax"]
    
    # track utility
    total_utility = 0.0
    
    discount_factor = 1.0
    
    alive = True
    
    for t in range(1, T+1):
        # 1) Survive check
        if np.random.rand() > survival_prob:
            # person dies at the start of year t
            alive = False
            # Compute bequest utility:
            bequest_amount = W_roth + W_tax
            bu = bequest_utility(bequest_amount, b=b, gamma=gamma)
            # discount that bequest at time t:
            discounted_bu = bu / ((1 + r_timepref)**t)
            total_utility += discounted_bu
            break
        
        # 2) Social Security inflow
        #    We treat it as a direct addition to consumption
        #    or as part of wealth. For simplicity, let's just 
        #    treat it as direct consumption.  But often we deposit 
        #    it into the checking account, then decide consumption from total.
        #    We'll do the latter to keep consistent with "all $ in portfolio".
        
        # Let's deposit into the taxable account (post-tax).
        # Actually we said it's "after tax" so let's just add it to W_tax:
        W_tax += social_security
        
        # 3) Decide consumption from total wealth
        total_wealth = W_roth + W_tax
        
        # We do a simple fraction-based policy: withdraw (consum_fraction * total_wealth)
        # (above the guaranteed social_security which we already added, so careful.)
        # Another approach is to define your policy to always consume = subsistence + fraction*(wealth - something).
        
        consumption_from_portfolio = consum_fraction * total_wealth
        
        # We can specify how we take from Roth vs Taxable. 
        # For simplicity, withdraw from taxable first up to consumption_from_portfolio.
        
        cons_left = consumption_from_portfolio
        if W_tax >= cons_left:
            W_tax -= cons_left
        else:
            shortfall = cons_left - W_tax
            W_tax = 0.0
            W_roth = max(W_roth - shortfall, 0.0)
        
        # Net consumption:
        c_t = consumption_from_portfolio  # plus the 30k SS, if you want. 
        # If you want the 30k SS to be part of consumption, then 
        # c_t += social_security
        # but that means you're double-counting the deposit. 
        # We'll keep it consistent: we deposit SS into W_tax, then withdraw from total.

        # 4) Compute immediate utility
        U_t = crra_utility(c_t, gamma=gamma, subsistence=subsistence, gamma_low=gamma_low)
        # discount the utility
        discounted_U_t = U_t / ((1 + r_timepref)**t)
        total_utility += discounted_U_t
        
        # 5) Invest remainder in safe vs. stock
        #    W_roth -> fraction stock_frac_roth in stock, remainder in safe
        #    W_tax -> fraction stock_frac_tax in stock, remainder in safe
        #    Then realize random returns:
        
        # For Roth, no tax on returns
        w_roth_stock = W_roth * stock_frac_roth
        w_roth_safe  = W_roth * (1 - stock_frac_roth)
        
        # For Taxable, tax on returns
        w_tax_stock = W_tax * stock_frac_tax
        w_tax_safe  = W_tax * (1 - stock_frac_tax)
        
        # 6) Realize returns for each portion
        # safe return is r_safe (no randomness)
        roth_safe_next = w_roth_safe * (1 + r_safe)
        
        # draw one random stock return:
        stock_ret = draw_lognormal_return(mean_stock, stdev_stock, n_sims=1)[0]
        
        roth_stock_next = w_roth_stock * (1 + stock_ret)  # no tax
        # total new Roth:
        W_roth = roth_safe_next + roth_stock_next
        
        # For taxable safe portion:
        tax_safe_next = w_tax_safe * (1 + r_safe)
        
        # For taxable stock portion (with immediate 20% tax on gains):
        # Gains = w_tax_stock * stock_ret
        # Tax on gains = 0.20 * Gains (if Gains>0, ignoring capital loss rules for simplicity)
        # W_tax_stock_next = principal + (1 - tax_rate)*gains
        gains = w_tax_stock * stock_ret
        # if you want to allow negative returns, which can happen with lognormal if <0 after shift
        # it complicates the model a bit. We'll just apply 20% to positive gains only for simplicity:
        if gains > 0:
            tax_on_gains = tax_rate * gains
        else:
            tax_on_gains = 0.0  # ignoring negative gains/loss offset for simplicity
        
        tax_stock_next = w_tax_stock + gains - tax_on_gains
        
        W_tax = tax_safe_next + tax_stock_next
        
        # 7) End of year => next loop
    
    # If you finish T years alive, you also get bequest utility at time T+1 (or not, 
    # depending on how you define it). We'll do it if you prefer:
    if alive:
        # final bequest:
        final_wealth = W_roth + W_tax
        bu = bequest_utility(final_wealth, b=b, gamma=gamma)
        discounted_bu = bu / ((1 + r_timepref)**(T+1))
        total_utility += discounted_bu
    
    return total_utility, W_roth + W_tax, alive


def run_monte_carlo(
    n_sims=10_000,
    **kwargs
):
    """
    Run many simulations and average the results.
    kwargs are forwarded to simulate_life_path.
    Returns average discounted utility, array of final wealth, fraction alive, etc.
    """
    utils = np.zeros(n_sims)
    final_wealths = np.zeros(n_sims)
    alive_counts = 0
    
    for i in range(n_sims):
        total_util, final_w, is_alive = simulate_life_path(**kwargs)
        utils[i] = total_util
        final_wealths[i] = final_w
        if is_alive:
            alive_counts += 1
    
    avg_util = np.mean(utils)
    avg_final_w = np.mean(final_wealths)
    alive_fraction = alive_counts / n_sims
    return avg_util, avg_final_w, alive_fraction


# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Example: run 10k simulations with default parameters
    # and a simple policy of 4% draw, 80% stock in Roth, 60% stock in Taxable.
    
    policy_dict = {
        "consum_fraction": 0.04,
        "stock_fraction_roth": 0.80,
        "stock_fraction_tax": 0.60
    }
    
    n_sims = 5000
    avg_util, avg_w, frac_alive = run_monte_carlo(
        n_sims=n_sims,
        policy_dict=policy_dict,
        survival_prob=0.97,
        T=35
    )
    print(f"After {n_sims} sims, average discounted utility = {avg_util:.2f}")
    print(f"Average final wealth = ${avg_w:,.2f}")
    print(f"Fraction surviving all 35 years = {frac_alive:.2%}")
