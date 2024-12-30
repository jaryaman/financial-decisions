import numpy as np
import polars as pl

from findec.utility import crra_utility
from findec.utility import wealth_to_gamma, bequest_utility
from findec.policy import policy
from findec.dataclasses import Preferences, Assets, State
from findec.returns import RiskyAsset
from findec.survival import (
    age_to_death_probability_female,
    age_to_death_probability_male,
)
from tqdm import tqdm


def simulate_life_paths(*args, n_sims: int, **kwargs) -> pl.DataFrame:
    dfs = []
    for i in tqdm(range(n_sims)):
        states = simulate_life_path(*args, **kwargs)
        df = pl.DataFrame([s.as_dict() for s in states.values()])
        df = df.with_columns(pl.lit(i, dtype=pl.Int64()).alias("run_number"))
        dfs.append(df)
    return pl.concat(dfs)


def simulate_life_path(
    *,
    expected_return_risky: float,
    std_dev_return_risky: float,
    risk_free_rate: float,
    pref: Preferences,
    a: Assets,
    social_security: float,
    time_horizon: int,  # maximum number of years we will live from current age. Can set this to very large numbers.
    rng_seed: int | None = None,
    starting_age: int = 65,
    is_male: bool = False,
    with_survival_probabilities: bool = True,
) -> dict[int, State]:
    if rng_seed is not None:
        np.random.seed(rng_seed)
    if is_male:
        age_to_death_probability = age_to_death_probability_male
    else:
        age_to_death_probability = age_to_death_probability_female

    expected_excess_return = expected_return_risky - risk_free_rate
    ra = RiskyAsset(
        expected_excess_return=expected_excess_return,
        standard_deviation=std_dev_return_risky,
        risk_free_rate=risk_free_rate,
    )

    total_utility = 0.0
    total_consumption = 0.0
    alive = True
    states = {
        starting_age: State(
            tax_free=a.tax_free,
            taxable=a.taxable,
            total_utility=total_utility,
            alive=alive,
            age=starting_age,
            consumption=None,
            consumption_fraction=None,
            risky_return=None,
        )
    }

    for t in range(1, time_horizon + 1):
        age = starting_age + t
        gamma = wealth_to_gamma(
            a.total_wealth,
            subsistence=pref.subsistence,
            gamma_below_subsistence=pref.gamma_below_subsistence,
            gamma_above_subsistence=pref.gamma_above_subsistence,
        )

        if (
            with_survival_probabilities
            and np.random.rand() < age_to_death_probability[age]
        ):
            alive = False
            bu = bequest_utility(a.total_wealth, b=pref.bequest_param, gamma=gamma)
            total_utility += bu
            states[age] = State(
                tax_free=a.tax_free,
                taxable=a.taxable,
                total_utility=total_utility,
                alive=alive,
                age=age,
                consumption=None,
                consumption_fraction=None,
                risky_return=None,
            )
            break

        # 1) Income from social security. Let's assume it has to go into the taxable account.
        a.taxable += social_security

        # 2) Decide policy
        pol = policy(
            time_horizon=time_horizon - t + 1,
            gamma=gamma,
            pref=pref,
            risk_free_rate=risk_free_rate,
            risky_asset=ra,
        )

        # 3) Use policy to decide how much to consume
        consumption_from_portfolio = a.consume(
            pol.consumption_fraction * a.total_wealth
        )
        total_consumption += consumption_from_portfolio

        # 4) Compute immediate utility from consumption
        utility_of_consumption = crra_utility(consumption_from_portfolio, gamma=gamma)
        discounted_utility_of_consumption = utility_of_consumption / (
            (1 + pref.rate_time_preference) ** t
        )
        total_utility += discounted_utility_of_consumption

        # 5) Invest remainder in safe/risky assets

        taxable_risky = pol.risky_asset_fraction_taxable * a.taxable
        taxable_safe = a.taxable - taxable_risky

        tax_free_risky = pol.risky_asset_fraction_tax_free * a.tax_free
        tax_free_safe = a.tax_free - tax_free_risky

        risky_returns = float(ra.draw())

        taxable_risky_next = taxable_risky * (1 + risky_returns)
        tax_free_risky_next = tax_free_risky * (1 + risky_returns)
        taxable_safe_next = taxable_safe * (1 + risk_free_rate)
        tax_free_safe_next = tax_free_safe * (1 + risk_free_rate)

        a.taxable = taxable_risky_next + taxable_safe_next
        a.tax_free = tax_free_risky_next + tax_free_safe_next

        states[age] = State(
            tax_free=a.tax_free,
            taxable=a.taxable,
            total_utility=total_utility,
            alive=alive,
            age=age,
            consumption=consumption_from_portfolio,
            consumption_fraction=pol.consumption_fraction,
            risky_return=risky_returns,
        )
        # End of year. Next loop.

    if alive:
        # final bequest
        bu = bequest_utility(a.total_wealth, b=pref.bequest_param, gamma=gamma)
        total_utility += bu

        states[age] = State(
            tax_free=a.tax_free,
            taxable=a.taxable,
            total_utility=total_utility,
            alive=alive,
            age=age,
            consumption=None,
            consumption_fraction=None,
            risky_return=None,
        )

    return states
