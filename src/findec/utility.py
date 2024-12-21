import numpy as np


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
    if gamma == 1.0:
        certainty_equivalent_final_wealth = np.exp(expected_utility)
    else:
        certainty_equivalent_final_wealth = (1 - (gamma - 1) * expected_utility) ** (
            1 / (1 - gamma)
        )
    certainty_equivalent_return = certainty_equivalent_final_wealth / initial_wealth - 1
    return certainty_equivalent_return
