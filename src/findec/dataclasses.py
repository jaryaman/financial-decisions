from dataclasses import dataclass, asdict
import copy


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
class Assets:
    tax_free: float
    taxable: float  # e.g. a stocks and shares ISA (Roth IRA in US)
    tax_rate: float  # e.g. capital gains on a stocks/shares account

    @property
    def total_wealth(self):
        return self.tax_free + self.taxable

    def consume(self, amount: float) -> float:
        """Consume first from the taxable account, then consume from the tax-free account.
        If you consume from the taxable account, you will incur a tax fee and consume less
        than the target amount of consumption."""
        if self.taxable > amount:
            self.taxable -= amount
            return amount * (1 - self.tax_rate)
        else:
            shortfall = amount - self.taxable
            consumed: float = self.taxable * (1 - self.tax_rate)
            self.taxable = 0
            if shortfall > self.tax_free:
                consumed += self.tax_free
                self.tax_free = 0
            else:
                consumed += shortfall
                self.tax_free -= shortfall
            return consumed


@dataclass(frozen=True)
class State:
    age: int
    alive: bool
    tax_free: float
    taxable: float
    risky_return: float | None
    consumption: float | None
    consumption_fraction: float | None
    total_utility: float
    annual_utility: float | None
    bequest: float | None

    def as_dict(self):
        return asdict(self)
