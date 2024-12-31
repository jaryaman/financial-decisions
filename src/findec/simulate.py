import numpy as np
import polars as pl
import copy

from findec.utility import crra_utility
from findec.utility import wealth_to_gamma, bequest_utility
from findec.policy import policy
from findec.dataclasses import Preferences, Assets, State
from findec.returns import RiskyAsset, DistributionType
from findec.survival import (
    age_to_death_probability_female,
    age_to_death_probability_male,
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
    pref: Preferences,
    a: Assets,
    social_security: float,
    time_horizon: int,  # maximum number of years we will live from current age. Can set this to very large numbers.
    rng_seed: int | None = None,
    rng_seed_offset: int | None = None,
    starting_age: int = 65,
    is_male: bool = False,
    with_survival_probabilities: bool = True,
    returns_distribution_type: DistributionType = DistributionType.NORMAL,
) -> dict[int, State]:
    if rng_seed_offset is not None and rng_seed is not None:
        np.random.seed(rng_seed_offset + rng_seed)
    if is_male:
        age_to_death_probability = age_to_death_probability_male
    else:
        age_to_death_probability = age_to_death_probability_female
    

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

    for t in range(1, time_horizon + 1):
        age = starting_age + t
        gamma = wealth_to_gamma(
            a.total_wealth_inflation_adjusted(t),
            subsistence=pref.subsistence,
            gamma_below_subsistence=pref.gamma_below_subsistence,
            gamma_above_subsistence=pref.gamma_above_subsistence,
        )

        if (
            with_survival_probabilities
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
        a.taxable += social_security

        # 2) Decide policy
        pol = policy(
            time_horizon=time_horizon - t + 1,
            bequest_param=pref.bequest_param,
            gamma=gamma,
            pref=pref,
            risk_free_rate=risk_free_rate,
            risky_asset=ra,
        )

        # 3) Use policy to decide how much to consume        

        desired_consumption_from_portfolio_pre_tax = pol.consumption_fraction * a.total_wealth
        actual_consumption_from_portfolio_post_tax = a.consume(
            desired_consumption_from_portfolio_pre_tax
        )       

        consumption_from_portfolio_post_tax_post_inflation = (
            actual_consumption_from_portfolio_post_tax * a.inflation_discount_factor(t)
        )
        total_consumption += consumption_from_portfolio_post_tax_post_inflation

        # 4) Compute immediate utility from consumption
        utility_of_consumption = crra_utility(
            consumption_from_portfolio_post_tax_post_inflation, gamma=gamma
        )
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
            portfolio_value_post_inflation=a.total_wealth_inflation_adjusted(t),
            total_utility=total_utility,
            total_consumption=total_consumption,
            alive=alive,
            age=age,
            desired_consumption_pre_tax=desired_consumption_from_portfolio_pre_tax,
            actual_consumption_post_tax=actual_consumption_from_portfolio_post_tax,
            consumption_post_tax_post_inflation=consumption_from_portfolio_post_tax_post_inflation,
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
            consumption_post_tax_post_inflation=consumption_from_portfolio_post_tax_post_inflation,
            consumption_fraction=pol.consumption_fraction,
            risky_return=risky_returns,
            annual_utility=bu + discounted_utility_of_consumption,
            bequest_post_inflation=a.total_wealth_inflation_adjusted(t),
        )

    return states
