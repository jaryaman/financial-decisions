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
    risky_asset_fraction_allocation = merton_share(
        expected_excess_return=risky_asset.expected_excess_return,
        std_dev_return=risky_asset.standard_deviation,
        gamma=gamma,
    )

    return_risk_adjusted_portfolio = risk_free_rate + risk_adjusted_excess_return(
        expected_excess_return=risky_asset.expected_excess_return,
        std_dev_return=risky_asset.standard_deviation,
        gamma=gamma,
        frac_risky_asset=risky_asset_fraction_allocation,
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
        risky_asset_fraction_tax_free=risky_asset_fraction_allocation,
        risky_asset_fraction_taxable=risky_asset_fraction_allocation,
    )
