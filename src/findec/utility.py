import numpy as np
from findec.dataclasses import Preferences


def crra_utility(w: np.ndarray | float, *, gamma: float, eps: float = 1e-8):
    """
    Interestingly, also known as the Box-Cox transformation in stats
    https://en.wikipedia.org/wiki/Isoelastic_utility
    """
    if w < eps:
        return -1e9
    if gamma == 1:
        return np.log(w)
    return (1 - w ** (1 - gamma)) / (gamma - 1)


def certainty_equivalent_return(
    *, initial_wealth: float, expected_utility: float, gamma: float
):
    """Equivalent to risk-adjusted return, by definition"""
    if gamma == 1.0:
        certainty_equivalent_final_wealth = np.exp(expected_utility)
    else:
        certainty_equivalent_final_wealth = (1 - (gamma - 1) * expected_utility) ** (
            1 / (1 - gamma)
        )
    certainty_equivalent_return = certainty_equivalent_final_wealth / initial_wealth - 1
    return certainty_equivalent_return


def get_matching_utility(
    *,
    subsistence: float,
    gamma_above_subsistence: float,
    gamma_below_subsistence: float,
):
    return crra_utility(subsistence, gamma=gamma_above_subsistence) - crra_utility(
        subsistence, gamma=gamma_below_subsistence
    )


def composite_crra_utility(
    w: float,
    *,
    pref: Preferences,
    matching_utility: float | None = None,
):
    """
    Piecewise CRRA:
    - If w < subsistence, treat as higher risk aversion gamma_subsistence.
    - If w < w_floor, it's all the same
    - Otherwise gamma as normal (e.g. 2).
    """
    if matching_utility is None:
        matching_utility = get_matching_utility(
            subsistence=pref.subsistence,
            gamma_above_subsistence=pref.gamma_above_subsistence,
            gamma_below_subsistence=pref.gamma_below_subsistence,
        )
    if w < pref.w_floor:
        return (
            crra_utility(pref.w_floor, gamma=pref.gamma_below_subsistence)
            + matching_utility
        )

    if w < pref.subsistence:
        return crra_utility(w, gamma=pref.gamma_below_subsistence) + matching_utility
    else:
        return crra_utility(w, gamma=pref.gamma_above_subsistence)


def bequest_utility(wealth, b=10, gamma=2.0):
    if wealth <= 0:
        return 0.0  # or negative utility, but typically 0 is fine if no wealth
    return b * (1 - (wealth / b) ** (1 - gamma)) / (gamma - 1)


def wealth_to_gamma(
    w: float,
    *,
    subsistence: float,
    gamma_below_subsistence: float,
    gamma_above_subsistence: float,
) -> float:
    if w < subsistence:
        return gamma_below_subsistence
    return gamma_above_subsistence
