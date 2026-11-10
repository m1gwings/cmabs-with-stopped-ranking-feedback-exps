"""EWS diagnostic export with exact PL references and Wilson intervals."""

import csv
from pathlib import Path

import numpy as np

from .comparison_diagnostic import exact_probabilities, wilson_interval

COLORS = ["blue!75!black", "orange!90!black", "teal!80!black", "violet!85!black"]
MARKS = ["*", "square*", "triangle*", "diamond*"]


class ComparisonFigures:
    def __init__(self, **settings):
        self.settings = settings

    def plot(self, summaries, figure, output_dir):
        if len(summaries) != 1 or summaries[0].count != 1:
            raise ValueError("Diagnostic expects one trial of coupled counts")
        settings = self.settings
        theta, pivot, item = settings["theta"], settings["pivot"], settings["item"]
        samples = settings["samples_per_outside_weight"]
        counts = iter(np.asarray(summaries[0].mean).reshape(-1, 2))
        rows = []
        for theta0 in settings["outside_weights"]:
            for cap in settings["caps"]:
                seen, wins = map(int, next(counts))
                exact_frequency, exact_mean = exact_probabilities(
                    theta, theta0, pivot, item, cap
                )
                fl, fu = wilson_interval(seen, samples, settings["confidence_level"])
                ml, mu = wilson_interval(wins, seen, settings["confidence_level"])
                rows.append(
                    {
                        "theta0": theta0,
                        "m": cap,
                        "samples": samples,
                        "comparisons": seen,
                        "wins": wins,
                        "frequency": seen / samples,
                        "frequency_ci_lower": fl,
                        "frequency_ci_upper": fu,
                        "conditional_mean": wins / seen if seen else 0.0,
                        "mean_ci_lower": ml,
                        "mean_ci_upper": mu,
                        "exact_frequency": exact_frequency,
                        "exact_conditional_mean": exact_mean,
                    }
                )
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "comparison_diagnostic.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        common = r"width=\linewidth,height=.74\linewidth,font=\small,grid=major,grid style={gray!20},xlabel={Outside weight $\theta_0$},xmin=0,xmax=1,xtick={0,.25,.5,.75,1}"
        frequency_lines = [
            "% Exact PL references and empirical pointwise 95% Wilson intervals.",
            r"\begin{tikzpicture}",
            r"\begin{axis}["
            + common
            + r",ylabel={Comparison frequency},ymin=0,ymax=1.04,legend pos=north east,legend style={font=\small,cells={anchor=west}}]",
        ]
        mean_lines = [
            "% Same colors/markers as the frequency panel; dashed black line is 1/3.",
            r"\begin{tikzpicture}",
            r"\begin{axis}["
            + common
            + r",ylabel={Conditional win fraction},ymin=0,ymax=1.04,extra y ticks={0.333333333333},extra y tick labels={$1/3$}]",
            r"\addplot[black,dashed,thick,domain=0:1,samples=2] {1/3};",
        ]
        for cap, color, mark in zip(self.settings["caps"], COLORS, MARKS):
            selected = sorted(
                [r for r in rows if int(r["m"]) == cap],
                key=lambda r: float(r["theta0"]),
            )
            exact = " ".join(
                f"({t:.12g},{exact_probabilities(theta, float(t), pivot, item, cap)[0]:.12g})"
                for t in np.linspace(0, 1, 101)
            )
            frequency_lines += [
                r"\addplot[" + color + r",thick,no marks] coordinates {" + exact + "};",
                r"\addlegendentry{$m=" + str(cap) + "$}",
            ]
            for lines, field, lower, upper in [
                (
                    frequency_lines,
                    "frequency",
                    "frequency_ci_lower",
                    "frequency_ci_upper",
                ),
                (mean_lines, "conditional_mean", "mean_ci_lower", "mean_ci_upper"),
            ]:
                coordinates = " ".join(
                    f"({float(r['theta0']):.12g},{float(r[field]):.12g}) += (0,{float(r[upper]) - float(r[field]):.12g}) -= (0,{float(r[field]) - float(r[lower]):.12g})"
                    for r in selected
                )
                lines.append(
                    r"\addplot["
                    + color
                    + r",only marks,mark="
                    + mark
                    + r",mark size=1.7pt,forget plot,error bars/.cd,y dir=both,y explicit] coordinates {"
                    + coordinates
                    + "};"
                )
        for name, lines in [
            ("comparison_frequency", frequency_lines),
            ("comparison_conditional_mean", mean_lines),
        ]:
            lines += [r"\end{axis}", r"\end{tikzpicture}", ""]
            (out / (name + ".tikz")).write_text("\n".join(lines))
        return [
            out / "comparison_diagnostic.csv",
            out / "comparison_frequency.tikz",
            out / "comparison_conditional_mean.tikz",
        ]
