from dataclasses import dataclass


@dataclass
class Assets:
    tax_free: float
    taxable: float  # e.g. a stocks and shares ISA (Roth IRA in US)    
    inflation_rate: float    

    def __post_init__(self):        
        # Assume no unrealized gains at the start
        self.taxable_basis = self.taxable

    def invest_in_taxable(self, amount: float):
        self.taxable += amount
        self.taxable_basis += amount  # type: ignore

    def invest_in_tax_free(self, amount: float):
        self.tax_free += amount

    @property
    def total_wealth(self) -> float:
        return self.tax_free + self.taxable

    def inflation_discount_factor(self, years_from_now: float) -> float:
        return (1 - self.inflation_rate) ** years_from_now

    def total_wealth_inflation_adjusted(self, years_from_now: float) -> float:
        return (self.tax_free + self.taxable) * self.inflation_discount_factor(
            years_from_now
        )

    def grow(
        self,
        *,
        risk_free_rate: float,
        risky_returns: float,
        risky_asset_fraction: float,
    ):
        taxable_risky = risky_asset_fraction * self.taxable
        taxable_safe = self.taxable - taxable_risky

        tax_free_risky = risky_asset_fraction * self.tax_free
        tax_free_safe = self.tax_free - tax_free_risky

        taxable_risky_next = taxable_risky * (1 + risky_returns)
        tax_free_risky_next = tax_free_risky * (1 + risky_returns)
        taxable_safe_next = taxable_safe * (1 + risk_free_rate)
        tax_free_safe_next = tax_free_safe * (1 + risk_free_rate)

        self.taxable = taxable_risky_next + taxable_safe_next
        # NB: We do not update the basis. Growth means we now have a taxable gain.

        self.tax_free = tax_free_risky_next + tax_free_safe_next
