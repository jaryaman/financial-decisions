import numpy as np


# Got this from chatGPT, but looks sensible. Think more later.
def draw_lognormal_return(mean_return=0.09, stdev=0.20, n_sims=1):
    # Draw from normal with mean=mean_return, stdev=stdev,
    # then do (1 + normal_draw).
    # But that can lead to negative returns. Let's do a direct lognormal approach:

    # We want to produce returns R >= -100%.
    # Usually we do: R = exp(X) - 1, where X ~ Normal(m, s^2).
    # E[R] = E[exp(X) - 1] = exp(m + s^2/2) - 1.

    # So set exp(m + s^2/2) - 1 = mean_return => m + s^2/2 = ln(1 + mean_return).

    sigma_log = np.sqrt(np.log(1 + (stdev**2 / (1 + mean_return) ** 2)))  # approximate
    mu_log = np.log(1 + mean_return) - 0.5 * sigma_log**2

    # Now draw X ~ Normal(mu_log, sigma_log^2), then R = exp(X)-1
    X = np.random.normal(loc=mu_log, scale=sigma_log, size=n_sims)
    R = np.exp(X) - 1.0
    return R


def risk_adjusted_excess_return(
    expected_excess_return: float,
    std_dev_return: float,
    gamma: float,
    frac_risky_asset: float,
):
    """Page 87 of Haghani & White"""
    return (
        frac_risky_asset * expected_excess_return
        - gamma * (frac_risky_asset * std_dev_return) ** 2 / 2
    )
