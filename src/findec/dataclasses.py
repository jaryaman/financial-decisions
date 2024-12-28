from dataclasses import dataclass

@dataclass
class Preferences:
    gamma: float = 2.0
    gamma_low: float = 5.0
    subsistence: float = 30_000.0
    w_floor: float = 1_000.0
    bequest_param: float = 10.0
    rate_time_preference: float = 0.02

@dataclass
class Policy:
    consumption_fraction: float 
    risky_asset_fraction_tax_free: float 
    risky_asset_fraction_taxable: float 

@dataclass
class RiskyAsset:
    expected_excess_return: float 
    standard_deviation: float