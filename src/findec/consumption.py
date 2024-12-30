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
