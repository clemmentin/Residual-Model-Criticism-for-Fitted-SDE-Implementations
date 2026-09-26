"""Simulation, score and plotting helpers for nonlinear weight selection."""
import csv
import numpy as np
from scipy.stats import binom, ks_2samp


def innovations(rng, count, horizon, *, strong=1.0, weak=0.0):
    """Residual arrays divided by kappa where appropriate; no fitted theta.

    weak is an independent Y diffusion loading divided by kappa.
    Constants H=1, h=1/2, a=sigma=1 are fixed in this example.
    """
    h, a, q = 0.5, 0.5, 0.625
    x = np.zeros(count)
    rx, ry, slope, variance = (np.empty((count, horizon)) for _ in range(4))
    for k in range(horizon):
        z1, z2, zy = rng.standard_normal((3, count))
        bar = a * x
        middle = bar + np.sqrt(h) * strong * z1
        end = a * middle + np.sqrt(h) * strong * z2
        rx[:, k] = end - a * a * x
        ry[:, k] = h * (np.tanh(middle) - np.tanh(bar)) + weak * zy
        derivative = 1.0 - np.tanh(bar) ** 2
        slope[:, k] = h * a * derivative / (1.0 + a * a)
        variance[:, k] = h ** 3 * derivative ** 2 / (1.0 + a * a)
        x = end
    return rx, ry, slope, variance


def score_at(coefficients, error):
    base, linear, quadratic = coefficients
    return base + 2.0 * error * linear + error * error * quadratic


def envelopes(coefficients, radius, center=0.0):
    base, linear, quadratic = coefficients
    vertex = np.divide(-linear, quadratic, out=np.zeros_like(base),
                       where=quadratic > 0)
    vertex = np.clip(vertex, center-radius, center+radius)
    lower = score_at(coefficients, vertex)
    upper = np.maximum(score_at(coefficients, center-radius),
                       score_at(coefficients, center+radius))
    return lower, upper


def cdf_gap(first, second):
    return float(ks_2samp(first, second, method="asymp").statistic)


def rank_rejection(target, reference, m=499, alpha=0.05):
    """Integrate an independent size-m reference sample using its empirical CDF."""
    f = np.searchsorted(np.sort(reference), target, side="left") / len(reference)
    k = int(np.floor(alpha * (m + 1)))
    return float(np.mean(binom.cdf(k - 1, m, 1.0 - f)))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (out / "summary.csv").open(encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if float(r["kappa"]) == 0.1]
    colors = {"full": "#28659A", "selected": "#D27831", "fixed": "#5C8962"}
    labels = {"full": "Full whitening", "selected": "CDF bound", "fixed": "Fixed weights"}
    if any(r['method']=='level' for r in rows):
        colors={"full":"#28659A","selected":"#D27831","level":"#985DAD","fixed":"#5C8962"}
        labels['level']='Bound at 5% level'
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), sharex=True)
    panels = [("gap", "Conditional score gap", 1.0),
              ("null_rejection_cv", "Rejection under the true model (%)", 100.0),
              ("strong_matched", "Strong-driver power at equal level (%)", 100.0),
              ("weak_matched", "Weak-direction power at equal level (%)", 100.0)]
    for ax, (key, title, factor) in zip(axes.flat, panels):
        for method in colors:
            group = sorted([r for r in rows if r["method"] == method],
                           key=lambda r: int(r["training_paths"]))
            x = [int(r["training_paths"]) for r in group]
            y = [factor*float(r[key]) for r in group]
            lo = [factor*float(r[key+"_min"]) for r in group]
            hi = [factor*float(r[key+"_max"]) for r in group]
            ax.plot(x, y, "o-", color=colors[method], lw=1.8, ms=4, label=labels[method])
            ax.fill_between(x, lo, hi, color=colors[method], alpha=0.12, linewidth=0)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xscale("log", base=4)
        ax.set_xticks(x, [str(value) for value in x])
        ax.grid(axis="y", color="#e3e6e8", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        if key == "null_rejection_cv":
            ax.axhline(5.0, ls="--", lw=0.8, color="#444444")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right")
    axes[1, 0].set_xlabel("Independent training transitions")
    axes[1, 1].set_xlabel("Independent training transitions")
    fig.suptitle("Weight selection in a nonlinear finite-grid model", x=0.07, ha="left", fontsize=15)
    fig.text(0.07, 0.012, "kappa = 0.1; K = 20; 64 fits. Lines: means; bands: fit ranges, not confidence intervals.\n"
             "Conditional-error tolerance: 0.02; training/pilot failure probability: at most 0.005.\n"
             "Null rejection uses a control variate. Equal-level power uses an independent true-null calibration sample.",
             fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.085, 1, 0.955))
    fig.savefig(out / "weight_selection.png", dpi=180)
    fig.savefig(out / "weight_selection.svg")
    plt.close(fig)
