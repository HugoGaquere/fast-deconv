"""Matplotlib helpers shared by all stages: paper styling + dual PNG/PDF output."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402


def fmt_size(n):
    """Compact image-size tick label: 2000 -> '2k', 1500 -> '1.5k', 800 -> '800'."""
    if n >= 1000:
        return f"{n / 1000:g}k"
    return str(int(round(n)))


def fmt_val(v):
    """Compact y-value label across metrics spanning very different scales
    (sub-1 ms/iter up to tens-of-thousands Mpix.iter/s)."""
    if v <= 0:
        return ""
    if v >= 1000:
        return f"{v / 1000:.3g}k"
    if v >= 1:
        return f"{v:.4g}"
    return f"{v:.3g}"


def unit_of(label):
    """Pull the bracketed unit out of a metric label ('wall time [s]' -> 's')."""
    if "[" in label and "]" in label:
        return label[label.index("[") + 1:label.index("]")]
    return ""


def annotate_max(ax, xmax, ymax, x_sizes, unit="") -> None:
    """Call out a panel's peak value (the log axis makes the tallest curve's exact
    height hard to read). Right-aligned when the peak sits at the largest size so
    the label grows into the panel instead of off the right edge."""
    txt = f"max={fmt_val(ymax)}" + (f" {unit}" if unit else "")
    ha = "right" if xmax >= max(x_sizes) else "center"
    ax.annotate(txt, xy=(xmax, ymax), xytext=(0, 4), textcoords="offset points",
                ha=ha, va="bottom", fontsize=6, fontweight="bold", clip_on=False,
                color="#1a1a1a",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6",
                          lw=0.5, alpha=0.85))


def scaling_axes(ax, x_sizes) -> None:
    """Label the axes of a log-log scaling panel the way the bigms poster figures
    do: tick the x-axis at exactly the swept image sizes (compact 2k/10k labels,
    no log minor ticks), and label the y-axis at the decade powers plus the 2/3/5
    minor ticks with compact values. Call after set_xscale/set_yscale('log')."""
    ax.xaxis.set_major_locator(mticker.FixedLocator(sorted(set(x_sizes))))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_size(round(x))))
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    yfmt = mticker.FuncFormatter(lambda y, _: fmt_val(y))
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2.0, 3.0, 5.0)))
    ax.yaxis.set_major_formatter(yfmt)
    ax.yaxis.set_minor_formatter(yfmt)
    # one tick-label size per axis (major and minor must match), and a smaller
    # axis-label font than the default style.
    ax.tick_params(axis="both", which="both", labelsize=5)
    ax.tick_params(axis="x", labelrotation=45)
    ax.xaxis.label.set_size(6)
    ax.yaxis.label.set_size(6)


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "legend.framealpha": 0.9,
        "figure.constrained_layout.use": True,
    })


def save_fig(fig, base, extra_artists=None) -> None:
    """Save as <base>.png (drafts) and <base>.pdf (SPIE wants vector figures).
    `extra_artists` (e.g. a legend anchored outside the axes) are passed to the
    tight bbox so they are not clipped."""
    base = str(base)
    kw = {"bbox_inches": "tight"}
    if extra_artists:
        kw["bbox_extra_artists"] = list(extra_artists)
    fig.savefig(base + ".png", **kw)
    fig.savefig(base + ".pdf", **kw)
    plt.close(fig)
