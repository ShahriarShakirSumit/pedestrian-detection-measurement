"""Set the two-up result tables as single lists for a single-column page.

make_tables.py writes the four 46-row tables as two halves side by side, which
suits a wide two-column page. On a single-column page the halves do not fit,
so each table becomes one list: header once, left-half rows, then right-half
rows. Row order is exactly the reading order of the two-up version and no cell
changes.

    python code/unfold_tables.py --tablesDir <dir written by make_tables.py>
"""
import argparse, pathlib, re

SPEC = re.compile(r"\\begin\{tabular\}\{@\{\}(.+?)@\{\\hspace\{[0-9.]+pt\}\}\s*(.+?)@\{\}\}")
TABLES = ("tab_coco_all", "tab_nightowls_all", "tab_llvip_all", "tab_inventory_main")


def unfold(p):
    s = p.read_text(encoding="utf-8")
    m = SPEC.search(s)
    left_spec, right_spec = m.group(1).strip(), m.group(2).strip()
    assert left_spec.replace(" ", "") == right_spec.replace(" ", ""), (p, left_spec, right_spec)
    ncol = len(re.findall(r"[lrc]", left_spec.replace(" ", "")))
    head, rest = s.split("\\toprule\n", 1)
    header, rest = rest.split("\\midrule\n", 1)
    rows_block, tail = rest.split("\\bottomrule\n", 1)

    def cells(line):
        c = [x.strip() for x in line.rstrip().removesuffix("\\\\").split("&")]
        assert len(c) == 2 * ncol, (p, len(c), line)
        return c[:ncol], c[ncol:]

    h_left, h_right = cells(header.strip())
    assert h_left == h_right, (p, h_left, h_right)
    left, right = [], []
    for line in rows_block.strip().splitlines():
        a, b = cells(line)
        left.append(a)
        if any(b):
            right.append(b)
    head = head.replace(m.group(0), "\\begin{tabular}{@{}" + left_spec + "@{}}")
    head = head.replace("\\begin{table*}", "\\begin{table}")
    head = head.replace(" and read down the left half then the right", "")
    tail = tail.replace("\\end{table*}", "\\end{table}")
    p.write_text(head + "\\toprule\n" + " & ".join(h_left) + " \\\\\n\\midrule\n"
                 + "".join(" & ".join(r) + " \\\\\n" for r in left + right)
                 + "\\bottomrule\n" + tail, encoding="utf-8")
    return len(left) + len(right), ncol


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tablesDir", required=True)
    a = ap.parse_args()
    for t in TABLES:
        n, c = unfold(pathlib.Path(a.tablesDir) / f"{t}.tex")
        print(f"{t}: {n} rows, {c} columns")
