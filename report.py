"""Generate honest Markdown and optional SVG from observed run artifacts only."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def valid(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def groups_of(rows):
    if not rows:
        raise ValueError("No completed runs; refusing an empty evidence report")
    groups, seen = defaultdict(list), set()
    for row in rows:
        key = (row["architecture"], row["curriculum"], row.get("loss_mode", "plain"), row["seed"])
        if key in seen:
            raise ValueError(f"Duplicate run: {key}")
        seen.add(key)
        count, mae = row["reappearance_count"], row["reappearance_location_mae_px"]
        if not valid(row["test_mse"]) or type(count) is not int or count < 0:
            raise ValueError(f"Invalid run metrics: {key}")
        if (count == 0) != (mae is None) or (mae is not None and not valid(mae)):
            raise ValueError(f"Undefined/inconsistent occlusion error: {key}")
        groups[key[:3]].append(row)
    return groups


def fmt(value):
    return f"{value:.4f}" if valid(value) else "NA"


def fmt_signed(value):
    """Format a finite contrast, which may legitimately be negative."""
    if type(value) in (int, float) and math.isfinite(value):
        return f"{value:.4f}"
    return "NA"


def markdown(rows, provenance=None):
    groups = groups_of(rows)
    lines = ["# Predictive Mirror — measured results", "",
             "**Exploratory engineering evidence; not proof of biological perception.**",
             "The primary outcome is true barrier-reappearance position MAE (px).",
             "NA means no eligible reappearance events, not zero error.", ""]
    if provenance is not None:
        lines += [f"Campaign digest: {provenance.get('sha256', 'missing')}; source revision: {provenance.get('source_revision', 'missing')}.", ""]
    lines += ["| Model | Curriculum | Loss | Independent seeds | Occlusion events | Mean MSE | Mean per-seed reappearance MAE |",
              "|---|---|---|---:|---:|---:|---:|"]
    for (architecture, curriculum, loss), runlist in sorted(groups.items()):
        valid_errors = [r["reappearance_location_mae_px"] for r in runlist if r["reappearance_count"]]
        lines.append(f"| {architecture} | {curriculum} | {loss} | {len(runlist)} | "
                     f"{sum(r['reappearance_count'] for r in runlist)} | "
                     f"{fmt(sum(r['test_mse'] for r in runlist)/len(runlist))} | "
                     f"{fmt(sum(valid_errors)/len(valid_errors) if valid_errors else None)} |")
    lines += ["", "## Paired curriculum contrast", ""]
    contrasts = 0
    for architecture, loss in sorted({(a, loss) for a, _, loss in groups}):
        if (architecture, "staged", loss) not in groups or (architecture, "shuffled", loss) not in groups:
            continue
        a = {r["seed"]: r for r in groups[(architecture, "staged", loss)]}
        b = {r["seed"]: r for r in groups[(architecture, "shuffled", loss)]}
        pairs = [(a[s], b[s]) for s in sorted(a.keys() & b.keys())]
        deltas = [x["reappearance_location_mae_px"] - y["reappearance_location_mae_px"]
                  for x, y in pairs if x["reappearance_count"] and y["reappearance_count"]]
        lines.append(f"- {architecture}, {loss}: {len(deltas)}/{len(pairs)} pairs with events; "
                     f"mean staged−shuffled MAE = {fmt_signed(sum(deltas)/len(deltas) if deltas else None)} px.")
        contrasts += 1
    if not contrasts:
        lines.append("No paired, eligible staged/shuffled outcomes.")
    lines += ["", "## Important limitations", "",
              "- Frame observations within a trajectory are not independent replicates.",
              "- Fewer than five independent seeds is exploratory; no statistical significance is asserted.",
              "- Whole-image MSE can reward predicting empty backgrounds while missing objects.",
              "- Model gains do not establish biological regionalization, plasticity or conscious perception.",
              "- Check image/source digests, event coverage, operator cost, and negative results before publication.", ""]
    return "\n".join(lines)


def feedback_svg(data):
    """Two independent horizontal scales; use only provided feedback data."""
    targets = (("Visible-motion MSE", "moving_object_still_camera", "mse"),
               ("Occlusion reappearance MAE (pixels)", "moving_object_moving_camera",
                "reappearance_location_mae_px"))
    labels = (("Intact", "intact_trained"), ("Feedback-free trained", "feedback_free_trained"),
              ("Cut only at test", "intact_cut_at_test"))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 440" role="img" aria-label="Measured feedback conditions">',
           '<rect width="820" height="440" fill="white"/>',
           '<text x="16" y="27" font-family="sans-serif" font-size="17">Feedback comparisons (descriptive)</text>']
    for i, (title, regime, field) in enumerate(targets):
        y0 = 60 + i * 185
        values = [data[regime][key].get(field) for _, key in labels]
        maximum = max([v for v in values if valid(v)] or [1.0])
        maximum = max(maximum, 1e-12)
        svg.append(f'<text x="16" y="{y0}" font-family="sans-serif" font-size="14">{title}</text>')
        for j, ((label, _), value) in enumerate(zip(labels, values)):
            y = y0 + 28 + j * 42
            svg.append(f'<text x="18" y="{y+14}" font-family="sans-serif" font-size="12">{label}</text>')
            if valid(value):
                svg.append(f'<rect x="235" y="{y}" width="{315 * value / maximum:.2f}" height="18" fill="#476d91"/>')
            svg.append(f'<text x="580" y="{y+14}" font-family="monospace" font-size="13">{fmt(value)}</text>')
    svg.extend(['<text x="16" y="426" font-family="sans-serif" font-size="11">Separate units and scales; NA means no events.</text>',
                '</svg>'])
    return "\n".join(svg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--feedback", help="Paired feedback-control JSON for the optional figure")
    parser.add_argument("--out", default="reports/results.md")
    parser.add_argument("--figure", default="reports/feedback.svg")
    args = parser.parse_args()
    root = Path(args.sweep)
    rows = json.loads((root / "runs.json").read_text())
    manifest = root / "campaign.json"
    provenance = json.loads(manifest.read_text()) if manifest.is_file() else None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(rows, provenance))
    print(f"Wrote report for {len(rows)} completed runs: {out}")
    if args.feedback:
        data = json.loads(Path(args.feedback).read_text())
        figure = Path(args.figure)
        figure.parent.mkdir(parents=True, exist_ok=True)
        figure.write_text(feedback_svg(data))
        print(f"Wrote actual-data SVG: {figure}")


if __name__ == "__main__":
    main()
