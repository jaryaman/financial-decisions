from dataclasses import dataclass, asdict


@dataclass
class Preferences:
    gamma_above_subsistence: float = 2.0
    gamma_below_subsistence: float = 5.0
    subsistence: float = 30_000.0
    w_floor: float = 1_000.0
    bequest_param: float = 10.0
    rate_time_preference: float = 0.02


@dataclass
class Policy:
    consumption_fraction: float
    risky_asset_fraction: float    


@dataclass(frozen=True)
class State:
    age: int
    alive: bool
    tax_free: float
    taxable: float
    taxable_basis: float
    portfolio_value_post_inflation: float | None
    risky_return: float | None
    desired_consumption_pre_tax: float | None
    actual_consumption_post_tax: float | None
    consumption_post_tax_post_inflation: float | None
    consumption_fraction: float | None    
    total_utility: float
    total_consumption: float
    annual_utility: float | None
    bequest_post_inflation: float | None

    def as_dict(self):
        return asdict(self)
