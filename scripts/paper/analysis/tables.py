"""Write a table as CSV + Markdown + a LaTeX tabular fragment (for \\input)."""
import csv
from pathlib import Path

_TEX_ESCAPES = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$"}


def _tex(s) -> str:
    out = str(s)
    for k, v in _TEX_ESCAPES.items():
        out = out.replace(k, v)
    return out


def write_table(headers, rows, base, title: str = "") -> None:
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)

    with open(base.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

    md = []
    if title:
        md += [f"# {title}", ""]
    md += ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    base.with_suffix(".md").write_text("\n".join(md) + "\n")

    lines = [f"\\begin{{tabular}}{{{'l' * len(headers)}}}", "\\hline",
             " & ".join(_tex(h) for h in headers) + r" \\", "\\hline"]
    lines += [" & ".join(_tex(c) for c in r) + r" \\" for r in rows]
    lines += ["\\hline", "\\end{tabular}"]
    base.with_suffix(".tex").write_text("\n".join(lines) + "\n")
