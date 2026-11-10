"""Final-regret CSV and native PGFPlots figure from retained EWS summaries."""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import t


class MNLFigures:
    """Match the conference's pointwise Student-t intervals over repetitions."""

    def __init__(self, confidence_level=0.95):
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must lie in (0, 1)")
        self.confidence_level = confidence_level

    def plot(self, summaries, figure, output_dir):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        for summary in summaries:
            if summary.uncertainty_kind != "standard_error":
                raise ValueError("Student-t intervals require standard errors")
            half_width = (
                t.ppf((1 + self.confidence_level) / 2, summary.count - 1)
                * float(summary.uncertainty[-1])
                if summary.count > 1
                else float("nan")
            )
            mean = float(summary.mean[-1])
            rows.append(
                (
                    summary.labels["algorithm.name"],
                    float(summary.labels["data.params.theta0"]),
                    int(summary.count),
                    mean,
                    half_width,
                )
            )
        rows.sort(key=lambda row: (row[0], row[1]))
        csv_path = output / "final_regret_vs_theta0.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ("algorithm", "theta0", "repetitions", "mean", "ci_half_width")
            )
            writer.writerows(rows)
        lines = [
            f"% Pointwise {100 * self.confidence_level:g}% Student-t confidence intervals.",
            r"\begin{tikzpicture}",
            r"\begin{axis}[width=\linewidth,height=0.75\linewidth,",
            r"xmode=log,ymode=log,xlabel={Outside weight $\theta_0$},",
            r"ylabel={Cumulative Top-$k$ Regret},grid=major,",
            r"legend style={font=\scriptsize,cells={anchor=west}}]",
        ]
        for name, style, label in (
            ("journal_uif", "blue,solid", r"UIF (real pivot)"),
            ("sg_adaptive", "orange,dashdotted", r"AOA-RB$_{\mathrm{PL}}$-Adaptive"),
            ("sg_no_choice", "teal,dashed", r"AOA-RB$_{\mathrm{PL}}$"),
        ):
            selected = [row for row in rows if row[0] == name]
            if not selected:
                continue
            coordinates = " ".join(
                f"({theta0:.12g},{mean:.12g}) +- (0,{half:.12g})"
                for _, theta0, _, mean, half_width in selected
                for half in [half_width if np.isfinite(half_width) else 0.0]
            )
            lines.append(
                rf"\addplot[{style},mark=*,error bars/.cd,y dir=both,y explicit]"
                rf" coordinates {{{coordinates}}};"
            )
            lines.append(r"\addlegendentry{" + label + "}")
        lines += [r"\end{axis}", r"\end{tikzpicture}", ""]
        tikz_path = output / "final_regret_vs_theta0.tikz"
        tikz_path.write_text("\n".join(lines))
        return [csv_path, tikz_path]


COLORS = ["blue!75!black", "orange!90!black", "teal!80!black", "violet!85!black"]


def coordinates(rows, xkey, ykey):
    return " ".join(
        f"({float(r[xkey]) / 1000000:.12g},{float(r[ykey]) / 1000:.12g})" for r in rows
    )


def stopped_figures(final, curves, source):
    out = source
    out.mkdir(parents=True, exist_ok=True)
    common = r"width=\linewidth,height=.72\linewidth,grid=major,grid style={gray!20},legend style={font=\small,cells={anchor=west}},tick label style={font=\small},label style={font=\small},scaled y ticks=false,ylabel={Cumulative regret ($\times 10^3$)}"
    lines = [
        "% Pointwise 95% Student-t confidence intervals over 20 independent runs.",
        r"\begin{tikzpicture}",
        r"\begin{axis}["
        + common
        + r",xlabel={Ranking cap $m$},xtick={1,2,3,4},legend pos=north east]",
    ]
    for theta, color in zip([0.0, 0.1, 0.3, 1.0], COLORS):
        selected = sorted(
            [r for r in final if float(r["data.params.theta0"]) == theta],
            key=lambda r: int(r["data.params.m"]),
        )
        coords = " ".join(
            f"({r['data.params.m']},{float(r['mean']) / 1000:.12g}) +- (0,{(float(r['ci95_upper']) - float(r['mean'])) / 1000:.12g})"
            for r in selected
        )
        lines += [
            r"\addplot["
            + color
            + r",thick,mark=*,error bars/.cd,y dir=both,y explicit] coordinates {"
            + coords
            + "};",
            r"\addlegendentry{$\theta_0=" + f"{theta:g}" + "$}",
        ]
    lines += [r"\end{axis}", r"\end{tikzpicture}", ""]
    (out / "final_regret_vs_m.tikz").write_text("\n".join(lines))
    lines = [
        "% Requires the PGFPlots fillbetween library.",
        r"\begin{tikzpicture}",
        r"\begin{axis}["
        + common
        + r",xlabel={Round $t$ ($\times 10^6$)},scaled x ticks=false,xtick={0,.25,.5,.75,1},xmin=0,xmax=1,ymin=0,legend pos=south east]",
    ]
    for m, color in zip([1, 2, 3, 4], COLORS):
        selected = sorted(
            [
                r
                for r in curves
                if int(r["data.params.m"]) == m
                and float(r["data.params.theta0"]) == 0.3
            ],
            key=lambda r: float(r["x"]),
        )
        for r in selected:
            r["lower"] = float(r["mean"]) - float(r["half_width"])
            r["upper"] = float(r["mean"]) + float(r["half_width"])
        for bound in ["lower", "upper"]:
            lines.append(
                r"\addplot[name path="
                + f"{bound}{m}"
                + r",draw=none,forget plot] coordinates {"
                + coordinates(selected, "x", bound)
                + "};"
            )
        lines += [
            r"\addplot["
            + color
            + r",fill opacity=.12,forget plot] fill between[of="
            + f"lower{m} and upper{m}"
            + "];",
            r"\addplot["
            + color
            + r",thick,no marks] coordinates {"
            + coordinates(selected, "x", "mean")
            + "};",
            r"\addlegendentry{$m=" + str(m) + "$}",
        ]
    lines += [r"\end{axis}", r"\end{tikzpicture}", ""]
    (out / "cumulative_regret_theta0_0_3.tikz").write_text("\n".join(lines))
    return out


class StoppedRankingFigures:
    """Stopped-ranking panels from EWS aggregates, with Student-t intervals."""

    def plot(self, summaries, figure, output_dir):
        curves, final = [], []
        for summary in summaries:
            if summary.uncertainty_kind != "standard_error" or summary.count < 2:
                raise ValueError(
                    "Stopped-ranking plots require repeated-run standard errors"
                )
            half = t.ppf(0.975, summary.count - 1) * summary.uncertainty
            rows = [
                dict(
                    summary.labels,
                    x=x,
                    mean=mean,
                    count=summary.count,
                    uncertainty=error,
                    half_width=width,
                    ci95_upper=mean + width,
                )
                for x, mean, error, width in zip(
                    summary.x, summary.mean, summary.uncertainty, half
                )
            ]
            curves.extend(rows)
            final.append(rows[-1])
        output = Path(output_dir)
        stopped_figures(final, curves, output)
        return [
            output / "final_regret_vs_m.tikz",
            output / "cumulative_regret_theta0_0_3.tikz",
        ]
