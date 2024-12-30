from dataclasses import dataclass


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
    risky_asset_fraction_tax_free: float
    risky_asset_fraction_taxable: float


@dataclass
class RiskyAsset:
    expected_excess_return: float
    standard_deviation: float


@dataclass
class Assets:
    tax_free: float = 400_000.0
    taxable: float = 600_000.0

    @property
    def total_wealth(self):
        return self.tax_free + self.taxable

    def consume(self, amount: float) -> float:
        """Consume first from the taxable account, then consume from the tax-free account"""
        if self.taxable > amount:            
            self.taxable -= amount
            return amount
        else:
            shortfall = amount - self.taxable            
            consumed: float = self.taxable
            self.taxable = 0
            if shortfall > self.tax_free:
                consumed += self.tax_free
                self.tax_free = 0
            else:
                consumed += shortfall
                self.tax_free -= shortfall
            return consumed
