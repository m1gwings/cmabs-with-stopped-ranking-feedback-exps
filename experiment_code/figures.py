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
