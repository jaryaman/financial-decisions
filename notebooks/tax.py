import numpy as np

def simulate_taxes_on_realized_gains(
    years=30,
    init_value=600_000,
    init_basis=600_000,    # assume no unrealized gains at the start
    stock_fraction=0.60,   # fraction in stocks
    mu=0.09,               # mean stock return
    sigma=0.20,            # volatility stock
    r_bond=0.04,           # safe asset return
    tax_rate=0.20,         # capital gains tax on realized gains
    draw_fraction=0.04     # fraction of account to withdraw each year for consumption
):
    """
    Simple example: track total value, cost basis, 
    pay tax only on realized gains when we withdraw (sell).
    Returns: arrays of (value, basis, consumption, tax_paid)
    """
    np.random.seed(123)
    
    value_tax = init_value
    basis_tax = init_basis
    
    val_history = []
    basis_history = []
    cons_history = []
    tax_history = []
    
    for t in range(1, years+1):
        # 1) Break down into stock vs. bond
        val_stock = stock_fraction * value_tax
        val_bond  = (1 - stock_fraction) * value_tax
        
        # 2) Realize random return on the stock portion 
        #    (lognormal, or simpler approach with normal for illustration)
        stock_return = np.random.normal(mu, sigma)  # e.g. arithmetic return
        val_stock_after = val_stock * (1 + stock_return)
        
        # 3) Bond portion grows at r_bond
        val_bond_after = val_bond * (1 + r_bond)
        
        # 4) Combine => total new market value
        new_value = val_stock_after + val_bond_after
        
        # 5) Decide how much to withdraw
        withdrawal = draw_fraction * new_value
        
        # 6) Realized gains fraction
        #    fraction of the portfolio sold = withdrawal / new_value
        frac_sold = withdrawal / new_value if new_value > 1e-9 else 0.0
        
        # realized gain = frac_sold * (new_value - basis_tax)
        realized_gain = frac_sold * (new_value - basis_tax)
        # if realized_gain < 0 => no capital gains tax in typical model 
        # (you might have capital loss that could offset other gains)
        # For simplicity, let's apply tax only if > 0
        if realized_gain > 0:
            tax_owed = tax_rate * realized_gain
        else:
            tax_owed = 0.0
        
        # 7) Reduce the withdrawal by the tax if you want to handle "who pays the tax"? 
        # Often, the tax is also paid from the proceeds of the sale. 
        # So your net consumption = withdrawal - tax_owed
        # or you might pay the tax from leftover portfolio. 
        net_consumption = withdrawal - tax_owed
        if net_consumption < 0:
            # theoretically can happen if taxes exceed the withdrawal, 
            # which is weird. Let's just set it to zero or handle it differently.
            net_consumption = 0.0
        
        # 8) The new portfolio value after withdrawing that amount
        #    and paying taxes from it:
        new_portfolio_value = new_value - withdrawal
        
        # 9) Update cost basis
        #    fraction not sold: (1 - frac_sold)
        #    so basis_tax also scales by (1 - frac_sold)
        #    If we re-invest or add new contributions, we would raise the basis again
        new_basis = basis_tax * (1 - frac_sold)
        # if there's leftover from rebalancing or new contributions, we add that to basis.
        # In this simplified example, no new contributions, so we skip that step.
        
        # 10) store results
        value_tax = new_portfolio_value
        basis_tax = new_basis
        
        val_history.append(value_tax)
        basis_history.append(basis_tax)
        cons_history.append(net_consumption)
        tax_history.append(tax_owed)
    
    return np.array(val_history), np.array(basis_history), np.array(cons_history), np.array(tax_history)

def main():
    vals, bases, consum, taxes = simulate_taxes_on_realized_gains()
    print("Year | Value   | Basis   | Consumption | Tax Paid")
    for i, (v, b, c, t) in enumerate(zip(vals, bases, consum, taxes), start=1):
        print(f"{i:4d} | {v:8.0f} | {b:8.0f} | {c:10.0f} | {t:8.0f}")

if __name__ == "__main__":
    main()
