import polars as pl
from polars import col
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

QUANTILES_DEFAULT = [0.25, 0.5, 0.75]


def quantile_lineplot(
    data: pl.DataFrame, *, x: str, y: str, quantiles: list[float] | None = None
) -> Axes:
    if quantiles is None:
        quantiles = QUANTILES_DEFAULT

    df_q = (
        data.group_by(x)
        .agg(col(y).quantile(q).alias(f"q_{q:.2f}") for q in quantiles)
        .sort(x)
    )

    lower_quantile = quantiles[0]
    central_quantile = quantiles[1]
    upper_quantile = quantiles[2]

    _, ax = plt.subplots()
    ax.plot(
        df_q[x],
        df_q[f"q_{central_quantile:.2f}"],
        #markers=True,
        color="black",
        label=f"q_{central_quantile:.2f}",
    )
    ax.fill_between(
        df_q[x],
        df_q[f"q_{lower_quantile:.2f}"],
        df_q[f"q_{upper_quantile:.2f}"],
        alpha=0.5,
        color="red",
        label=f"[q_{lower_quantile:.2f}--q_{upper_quantile:.2f}]",
    )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()

    return ax
