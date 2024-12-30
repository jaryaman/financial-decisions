def optimal_consumption_infinite_horizon(
    return_risk_adjusted: float, rate_time_preference: float, gamma: float
):
    return return_risk_adjusted - (return_risk_adjusted - rate_time_preference) / gamma


def optimal_consumption_finite_horizon(
    return_risk_adjusted: float,
    rate_time_preference: float,
    gamma: float,
    time_horizon: float | int,
):
    c_infty = optimal_consumption_infinite_horizon(
        return_risk_adjusted, rate_time_preference, gamma
    )
    try:
        return c_infty / (1 - (1 + c_infty) ** (-time_horizon))
    except ZeroDivisionError as e:
        raise e
