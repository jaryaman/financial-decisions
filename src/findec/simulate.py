import numpy as np
import polars as pl
import copy

from findec.assets import Assets
from findec.utility import crra_utility
from findec.utility import wealth_to_gamma, bequest_utility
from findec.policy import policy
from findec.dataclasses import Preferences, State
from findec.returns import RiskyAsset, DistributionType
from findec.consumption import consume_from_assets
from findec.survival import (
    age_to_death_probability_female,
    age_to_death_probability_male,
    age_to_life_expectancy_male,
    age_to_life_expectancy_female,
)
from tqdm import tqdm


def simulate_life_paths(*args, n_sims: int, **kwargs) -> pl.DataFrame:
    dfs = []
    for i in tqdm(range(n_sims)):
        copied_args = copy.deepcopy(args)
        copied_kwargs = copy.deepcopy(kwargs)
        states = simulate_life_path(rng_seed=i, *copied_args, **copied_kwargs)
        df = pl.DataFrame([s.as_dict() for s in states.values()])
        df = df.with_columns(pl.lit(i, dtype=pl.Utf8()).alias("run_number"))
        dfs.append(df)
    return pl.concat(dfs)


def simulate_life_path(
    *,
    expected_return_risky: float,
    std_dev_return_risky: float,
    risk_free_rate: float,
    tax_rate: float,
    pref: Preferences,
    a: Assets,
    social_security: float,
    time_horizon_max: int,  # maximum number of years we will live from current age. Can set this to very large numbers.
    rng_seed: int | None = None,
    rng_seed_offset: int | None = None,
    starting_age: int = 65,
    is_male: bool = False,
    with_longevity_uncertainty: bool = True,
    returns_distribution_type: DistributionType = DistributionType.NORMAL,
) -> dict[int, State]:
    if rng_seed_offset is not None and rng_seed is not None:
        np.random.seed(rng_seed_offset + rng_seed)

    if is_male:
        age_to_death_probability = age_to_death_probability_male
        age_to_life_expectancy = age_to_life_expectancy_male
    else:
        age_to_death_probability = age_to_death_probability_female
        age_to_life_expectancy = age_to_life_expectancy_female

    ra = RiskyAsset(
        expected_return=expected_return_risky,
        standard_deviation=std_dev_return_risky,
        distribution_type=returns_distribution_type,
    )

    total_utility = 0.0
    total_consumption = 0.0
    alive = True
    states = {
        starting_age: State(
            tax_free=a.tax_free,
            taxable=a.taxable,
            portfolio_value_post_inflation=a.total_wealth_inflation_adjusted(0),
            total_utility=total_utility,
            total_consumption=total_consumption,
            alive=alive,
            age=starting_age,
            desired_consumption_pre_tax=None,
            actual_consumption_post_tax=None,
            consumption_fraction=None,
            consumption_post_tax_post_inflation=None,
            risky_return=None,
            annual_utility=None,
            bequest_post_inflation=None,
        )
    }

    for t in range(1, time_horizon_max + 1):
        age = starting_age + t
        gamma = wealth_to_gamma(
            a.total_wealth_inflation_adjusted(t),
            subsistence=pref.subsistence,
            gamma_below_subsistence=pref.gamma_below_subsistence,
            gamma_above_subsistence=pref.gamma_above_subsistence,
        )

        if (
            with_longevity_uncertainty
            and np.random.rand() < age_to_death_probability[age]
        ):  # He's dead, Jim.
            alive = False
            bu = bequest_utility(
                a.total_wealth_inflation_adjusted(t), b=pref.bequest_param, gamma=gamma
            ) / ((1 + pref.rate_time_preference) ** t)
            total_utility += bu
            states[age] = State(
                tax_free=a.tax_free,
                taxable=a.taxable,
                total_utility=total_utility,
                total_consumption=total_consumption,
                alive=alive,
                age=age,
                desired_consumption_pre_tax=None,
                actual_consumption_post_tax=None,
                consumption_post_tax_post_inflation=None,
                consumption_fraction=None,
                portfolio_value_post_inflation=a.total_wealth_inflation_adjusted(t),
                risky_return=None,
                annual_utility=bu,
                bequest_post_inflation=a.total_wealth_inflation_adjusted(t),
            )
            break

        # 1) Income from social security. Let's assume it has to go into the taxable account.
        a.invest_in_taxable(social_security)

        time_horizon = (
            age_to_life_expectancy[age]
            if with_longevity_uncertainty
            else time_horizon_max - t + 1
        )

        # 2) Decide policy
        pol = policy(
            time_horizon=time_horizon,
            bequest_param=pref.bequest_param,
            gamma=gamma,
            pref=pref,
            risk_free_rate=risk_free_rate,
            risky_asset=ra,
        )

        # 3) Grow assets
        risky_returns = float(ra.draw())
        a.grow(
            risk_free_rate=risk_free_rate,
            risky_returns=risky_returns,
            risky_asset_fraction=pol.risky_asset_fraction,
        )

        # 4) Use policy to decide how much to consume
        desired_consumption_from_portfolio_pre_tax = (
            pol.consumption_fraction * a.total_wealth
        )

        # 5) Consume from assets
        actual_consumption_from_portfolio_post_tax = consume_from_assets(
            fractional_consumption=pol.consumption_fraction, assets=a, tax_rate=tax_rate
        )

        actual_consumption_from_portfolio_post_tax_post_inflation = (
            actual_consumption_from_portfolio_post_tax * a.inflation_discount_factor(t)
        )
        total_consumption += actual_consumption_from_portfolio_post_tax_post_inflation

        # 4) Compute immediate utility from consumption
        utility_of_consumption = crra_utility(
            actual_consumption_from_portfolio_post_tax_post_inflation, gamma=gamma
        )

        discounted_utility_of_consumption = utility_of_consumption / (
            (1 + pref.rate_time_preference) ** t
        )
        total_utility += discounted_utility_of_consumption

        # 5) Store state
        states[age] = State(
            tax_free=a.tax_free,
            taxable=a.taxable,
            portfolio_value_post_inflation=a.total_wealth_inflation_adjusted(t),
            total_utility=total_utility,
            total_consumption=total_consumption,
            alive=alive,
            age=age,
            desired_consumption_pre_tax=desired_consumption_from_portfolio_pre_tax,
            actual_consumption_post_tax=actual_consumption_from_portfolio_post_tax,
            consumption_post_tax_post_inflation=actual_consumption_from_portfolio_post_tax_post_inflation,
            consumption_fraction=pol.consumption_fraction,
            risky_return=risky_returns,
            annual_utility=discounted_utility_of_consumption,
            bequest_post_inflation=None,
        )
        # End of year. Next loop.

    if alive:
        # final bequest
        bu = bequest_utility(
            a.total_wealth_inflation_adjusted(t), b=pref.bequest_param, gamma=gamma
        ) / ((1 + pref.rate_time_preference) ** t)
        total_utility += bu

        states[age] = State(
            tax_free=a.tax_free,
            taxable=a.taxable,
            portfolio_value_post_inflation=a.total_wealth_inflation_adjusted(t),
            total_utility=total_utility,
            total_consumption=total_consumption,
            alive=alive,
            age=age,
            desired_consumption_pre_tax=desired_consumption_from_portfolio_pre_tax,
            actual_consumption_post_tax=actual_consumption_from_portfolio_post_tax,
            consumption_post_tax_post_inflation=actual_consumption_from_portfolio_post_tax_post_inflation,
            consumption_fraction=pol.consumption_fraction,
            risky_return=risky_returns,
            annual_utility=bu + discounted_utility_of_consumption,
            bequest_post_inflation=a.total_wealth_inflation_adjusted(t),
        )

    return states
