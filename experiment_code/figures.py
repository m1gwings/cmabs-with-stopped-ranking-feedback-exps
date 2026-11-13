"""Final-regret CSV and native PGFPlots figure from retained EWS summaries."""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import t

from .data import exact_probabilities, wilson_interval


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


def stopped_figures(final, source):
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
    return out


class StoppedRankingFigures:
    """Final stopped-ranking regret from EWS aggregates, with Student-t intervals."""

    def plot(self, summaries, figure, output_dir):
        final = []
        for summary in summaries:
            if summary.uncertainty_kind != "standard_error" or summary.count < 2:
                raise ValueError(
                    "Stopped-ranking plots require repeated-run standard errors"
                )
            mean = float(summary.mean[-1])
            half = t.ppf(0.975, summary.count - 1) * float(summary.uncertainty[-1])
            final.append(dict(summary.labels, mean=mean, ci95_upper=mean + half))
        output = Path(output_dir)
        stopped_figures(final, output)
        return [output / "final_regret_vs_m.tikz"]


class PaperFigures:
    """Native manuscript panels with pointwise confidence intervals for the mean.

    Fragments inherit the enclosing manuscript's font; input them directly
    inside the existing minipages, without scaling an embedded PDF.
    """

    def __init__(self, confidence_level=0.95):
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one")
        self.confidence_level = confidence_level

    def _half_width(self, summary):
        if summary.uncertainty_kind != "standard_error":
            raise ValueError("Confidence intervals require standard-error aggregates")
        if summary.count < 2:
            return np.full_like(summary.uncertainty, np.nan, dtype=float)
        critical = t.ppf((1 + self.confidence_level) / 2, df=summary.count - 1)
        return critical * summary.uncertainty

    def plot(self, summaries, figure, output_dir):
        output = Path(output_dir)
        files = []
        for group, filename in (
            ("comparison", "cumulative_regret"),
            ("feedback", "final_regret_vs_m"),
        ):
            selected = [s for s in summaries if s.labels["group"] == group]
            if not selected:
                continue
            lines = [
                f"% Pointwise {100 * self.confidence_level:g}% Student-t confidence intervals for the mean.",
                r"\begin{tikzpicture}",
                r"\definecolor{paperblue}{HTML}{1F77B4}",
                r"\definecolor{paperorange}{HTML}{FF7F0E}",
                r"\definecolor{papergreen}{HTML}{2CA02C}",
            ]
            options = [
                # Fix the inner axes as well as the outer canvas.  Letting PGFPlots
                # infer the box from tick labels/legends misaligns log/linear panels.
                "scale only axis",
                r"at={(0.17\linewidth,0.14\linewidth)}",
                "anchor=south west",
                r"width=0.80\linewidth",
                r"height=0.60\linewidth",
                r"font=\fontsize{4.5}{5.5}\selectfont",
                r"label style={font=\fontsize{5.5}{6.5}\selectfont}",
                "xlabel style={at={(axis description cs:0.5,-0.16)},anchor=north}",
                "ylabel style={at={(axis description cs:-0.16,0.5)},anchor=south}",
                "grid=major",
                "grid style={gray!35,line width=0.25pt}",
                "axis line style={line width=0.4pt}",
                "tick style={line width=0.35pt}",
                "tick align=outside",
                "major tick length=1.5pt",
                "minor tick length=0.8pt",
                "xtick pos=left",
                "ytick pos=left",
                r"ylabel={Cumulative Top-$k$ Regret}",
                r"legend style={font=\fontsize{4.2}{5.2}\selectfont,draw=gray!35,fill=white,cells={anchor=west,inner xsep=0.5pt,inner ysep=0.5pt},inner xsep=1pt,inner ysep=0.5pt,row sep=0pt}",
                "legend image post style={xscale=0.5}",
            ]
            if group == "comparison":
                horizon = max(float(s.x[-1]) for s in selected)
                exponent = int(np.floor(np.log10(horizon))) if horizon >= 1 else 0
                tick_positions = np.linspace(0, horizon, 6)
                tick_labels = tick_positions / 10**exponent
                options += [
                    rf"xlabel={{Learning Horizon ($10^{{{exponent}}}$)}}",
                    "xmin=0",
                    f"xmax={horizon:.12g}",
                    "ymin=0",
                    "scaled x ticks=false",
                    "xtick={" + ",".join(f"{v:.12g}" for v in tick_positions) + "}",
                    "xticklabels={" + ",".join(f"{v:g}" for v in tick_labels) + "}",
                    "scaled y ticks=false",
                    "yticklabel style={/pgf/number format/1000 sep={}}",
                    "legend style={at={(0.98,0.2)},anchor=east}",
                ]
                series = [
                    (s.labels["algorithm.name"], s.x, s.mean, self._half_width(s))
                    for s in selected
                ]
                if max(float(np.max(s.mean)) for s in selected) >= 10000:
                    options += [
                        "scaled y ticks=base 10:-3",
                        "ytick scale label code/.code={}",
                        r"ylabel={Cumulative Top-$k$ Regret ($10^3$)}",
                    ]
            else:
                key = "data.params.m"
                values = sorted({s.labels[key] for s in selected})
                options += [
                    r"xlabel={Feedback Level $m$}",
                    "xtick={" + ",".join(f"{v:.12g}" for v in values) + "}",
                    "ymin=0",
                    "scaled y ticks=false",
                    "yticklabel style={/pgf/number format/1000 sep={}}",
                ]
                series = []
                for name in sorted({s.labels["algorithm.name"] for s in selected}):
                    rows = sorted(
                        [s for s in selected if s.labels["algorithm.name"] == name],
                        key=lambda s: s.labels[key],
                    )
                    series.append(
                        (
                            name,
                            np.array([s.labels[key] for s in rows]),
                            np.array([s.mean[-1] for s in rows]),
                            np.array([self._half_width(s)[-1] for s in rows]),
                        )
                    )
            if (
                group == "feedback"
                and max(float(s.mean[-1]) for s in selected) >= 10000
            ):
                options += [
                    "scaled y ticks=base 10:-3",
                    "ytick scale label code/.code={}",
                    r"ylabel={Cumulative Top-$k$ Regret ($10^3$)}",
                ]
            lines.append(r"\begin{axis}[" + ",\n".join(options) + "]")
            styles = {
                "topk_uif": ("paperblue,solid", r"\textsc{Top-$k$-UIF}"),
                "rec_maxmin_ucb": ("paperorange,dashed", r"\textsc{RecMaxMinUCB}"),
            }
            # Match the manuscript's legend ordering.
            series.sort(key=lambda row: list(styles).index(row[0]))
            for index, (name, x, mean, half_width) in enumerate(series):
                style, label = styles[name]
                if group == "comparison":
                    if np.isfinite(half_width).all():
                        for suffix, values in (
                            ("lo", mean - half_width),
                            ("hi", mean + half_width),
                        ):
                            coords = " ".join(
                                f"({a:.12g},{b:.12g})" for a, b in zip(x, values)
                            )
                            lines.append(
                                rf"\addplot[draw=none,name path=c{index}{suffix},forget plot] coordinates {{{coords}}};"
                            )
                        color = style.split(",")[0]
                        lines.append(
                            rf"\addplot[{color},fill opacity=0.18,draw=none,forget plot] fill between[of=c{index}lo and c{index}hi];"
                        )
                    coords = " ".join(f"({a:.12g},{b:.12g})" for a, b in zip(x, mean))
                    lines.append(
                        rf"\addplot[{style},line width=0.6pt] coordinates {{{coords}}};"
                    )
                else:
                    coords = " ".join(
                        f"({a:.12g},{b:.12g}) +- (0,{e:.12g})"
                        for a, b, e in zip(x, mean, np.nan_to_num(half_width))
                    )
                    lines.append(
                        rf"\addplot[{style},line width=0.6pt,mark=*,mark size=0.85pt,error bars/.cd,y dir=both,y explicit,error mark options={{mark size=1pt,line width=0.4pt}}] coordinates {{{coords}}};"
                    )
                if group != "feedback":
                    lines.append(r"\addlegendentry{" + label + "}")
            lines += [
                r"\end{axis}",
                r"\pgfresetboundingbox",
                r"\path[use as bounding box] (0,0) rectangle (\linewidth,0.75\linewidth);",
                r"\end{tikzpicture}",
                "",
            ]
            path = output / (filename + ".tikz")
            path.write_text("\n".join(lines))
            files.append(path)
        return files


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
