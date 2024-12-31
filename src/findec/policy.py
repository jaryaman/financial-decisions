from findec.consumption import optimal_consumption_finite_horizon
from findec.returns import risk_adjusted_excess_return, RiskyAsset
from findec.dataclasses import Preferences, Policy


def merton_share(*, expected_excess_return: float, gamma: float, std_dev_return: float):
    """Optimal frac_risky_assert for risky asset"""
    return expected_excess_return / (gamma * std_dev_return**2)


def policy(
    *,
    time_horizon: float | int,
    gamma: float,
    pref: Preferences,
    risk_free_rate: float,
    risky_asset: RiskyAsset,
    bequest_param: float | None,
) -> Policy:
    mu = risky_asset.expected_return - risk_free_rate
    sigma = risky_asset.standard_deviation
    k = merton_share(
        expected_excess_return=mu,
        std_dev_return=sigma,
        gamma=gamma,
    )

    return_risk_adjusted_portfolio = risk_free_rate + risk_adjusted_excess_return(
        expected_excess_return=mu,
        std_dev_return=sigma,
        gamma=gamma,
        frac_risky_asset=k,
    )

    consumption_fraction = optimal_consumption_finite_horizon(
        return_risk_adjusted=return_risk_adjusted_portfolio,
        rate_time_preference=pref.rate_time_preference,
        gamma=gamma,
        time_horizon=time_horizon,
        bequest_param=bequest_param,
    )

    return Policy(
        consumption_fraction=consumption_fraction,
        risky_asset_fraction_tax_free=k,
        risky_asset_fraction_taxable=k,
    )
