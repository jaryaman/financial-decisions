from findec.assets import Assets


def optimal_consumption_infinite_horizon(
    return_risk_adjusted: float, rate_time_preference: float, gamma: float
):
    return return_risk_adjusted - (return_risk_adjusted - rate_time_preference) / gamma


def optimal_consumption_finite_horizon(
    return_risk_adjusted: float,
    rate_time_preference: float,
    gamma: float,
    time_horizon: float | int,
    bequest_param: float | None,
):
    c_infty = optimal_consumption_infinite_horizon(
        return_risk_adjusted, rate_time_preference, gamma
    )
    if time_horizon == 0:
        raise ValueError("time_horizon = 0 implies infinite consumption!")

    if bequest_param is not None:
        """Although not written explicitly, this is how I interpret the footnote on p138 of
        Haghani & White
        """
        time_horizon += bequest_param

    return c_infty / (1 - (1 + c_infty) ** (-time_horizon))


def consume_from_taxable_assets(
    *, withdrawal: float, assets: Assets, tax_rate: float
) -> float:
    initial_taxable_assets = assets.taxable

    frac_sold = withdrawal / assets.taxable
    realized_gain = frac_sold * (assets.taxable - assets.taxable_basis)

    # if realized_gain < 0 => no capital gains tax in typical model
    # (you might have capital loss that could offset other gains)
    # For simplicity, let's apply tax only if > 0
    if realized_gain > 0.0:
        tax_owed = tax_rate * realized_gain
    else:
        tax_owed = 0.0

    net_consumption = withdrawal - tax_owed

    assets.taxable -= withdrawal
    assets.taxable_basis = assets.taxable_basis * (1 - frac_sold)

    if abs(initial_taxable_assets - net_consumption - tax_owed) > 1e-8:
        raise ValueError(
            f"Money not conserved! initial_taxable_assets={initial_taxable_assets}; net_consumption={net_consumption}; tax_owed={tax_owed}"
        )

    return net_consumption


def consume_from_assets(
    *, fractional_consumption: float, assets: Assets, tax_rate: float
) -> float:
    """Consume first from the taxable account, then consume from the tax-free account.
    If you consume from the taxable account, you will incur a tax fee and consume less
    than the target amount of consumption."""

    withdrawal = fractional_consumption * assets.total_wealth

    # Prefer to withdraw from taxable account first
    if assets.taxable > withdrawal:
        net_consumption = consume_from_taxable_assets(
            withdrawal=withdrawal, assets=assets, tax_rate=tax_rate
        )

    else:
        net_consumption = consume_from_taxable_assets(
            withdrawal=assets.taxable, assets=assets, tax_rate=tax_rate
        )

        shortfall = withdrawal - net_consumption
        if shortfall > assets.tax_free:
            net_consumption += assets.tax_free
            assets.tax_free = 0
        else:
            net_consumption += shortfall
            assets.tax_free -= shortfall

    return net_consumption
