from findec.consumption import optimal_consumption_finite_horizon
from findec.utility import wealth_to_gamma, merton_share
from findec.dataclasses import Preferences, Policy, RiskyAsset


def policy(
    time_horizon: float | int,
    w: float,
    pref: Preferences,
    return_risk_adjusted_portfolio: float,
    risky_asset: RiskyAsset,
) -> Policy:
    this_gamma = wealth_to_gamma(
        w, subsistence=pref.subsistence, gamma_low=pref.gamma_low, gamma=pref.gamma
    )
    consumption_fraction = optimal_consumption_finite_horizon(
        return_risk_adjusted=return_risk_adjusted_portfolio,
        rate_time_preference=pref.rate_time_preference,
        gamma=this_gamma,
        time_horizon=time_horizon,
    )

    risky_asset_fraction_tax_free = merton_share(risky_asset, this_gamma)
    risky_asset_fraction_taxable = merton_share(risky_asset, this_gamma)

    return Policy(
        consumption_fraction=consumption_fraction,
        risky_asset_fraction_tax_free=risky_asset_fraction_tax_free,
        risky_asset_fraction_taxable=risky_asset_fraction_taxable,
    )
