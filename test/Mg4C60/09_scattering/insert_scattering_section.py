#!/usr/bin/env python3
"""Insert the Mg4C60 mode-resolved scattering section into an existing report.html.

Usage:
    python insert_scattering_section.py /path/to/Mg4C60_kl_report          # dry run
    python insert_scattering_section.py /path/to/Mg4C60_kl_report --write

What --write does:
  * copies fig_scattering_c3_3p50.png (and c3_3p00 if present) into <report>/figures/
  * backs up report.html to report.html.bak
  * replaces the block that starts at the first <h2> whose text contains one of
    the markers ("声子散射", "散射", "scattering") and ends right before the next
    <h2>, with the content of scattering_section.html
  * if no marker is found, inserts the section just before </body>

Run it from the directory that contains scattering_section.html and the figure
(i.e. this folder), or pass --assets.
"""
import argparse, os, re, shutil, sys

MARKERS = ("声子散射", "散射", "scattering")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report_dir")
    ap.add_argument("--write", action="store_true", help="actually modify report.html")
    ap.add_argument("--assets", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--section", default="scattering_section.html")
    args = ap.parse_args()

    rep = os.path.abspath(args.report_dir)
    html_path = os.path.join(rep, "report.html")
    if not os.path.exists(html_path):
        sys.exit("no report.html in %s" % rep)

    section = open(os.path.join(args.assets, args.section)).read().strip()
    src = open(html_path).read()

    # locate the marker heading and the next <h2>
    pat = re.compile(r"<h2[^>]*>.*?(?:%s).*?</h2>" % "|".join(map(re.escape, MARKERS)),
                     re.IGNORECASE | re.DOTALL)
    m = pat.search(src)
    if m:
        start = m.start()
        nxt = re.search(r"<h2", src[m.end():])
        end = m.end() + nxt.start() if nxt else src.find("</body>")
        print("[insert] replacing existing scattering section: chars %d..%d" % (start, end))
        new = src[:start] + section + "\n" + src[end:]
    else:
        pos = src.find("</body>")
        if pos < 0:
            pos = len(src)
        print("[insert] no marker found -> inserting before </body> (char %d)" % pos)
        new = src[:pos] + section + "\n" + src[pos:]

    figdir = os.path.join(rep, "figures")
    figs = [f for f in os.listdir(args.assets) if f.startswith("fig_scattering_c3_") and f.endswith(".png")]
    for f in figs:
        print("[insert] figure ->", os.path.join("figures", f))

    if not args.write:
        print("[dry-run] nothing written. Re-run with --write to apply.")
        return

    os.makedirs(figdir, exist_ok=True)
    for f in figs:
        shutil.copy(os.path.join(args.assets, f), os.path.join(figdir, f))
    shutil.copy(html_path, html_path + ".bak")
    open(html_path, "w").write(new)
    print("[insert] report.html updated (backup: report.html.bak)")


if __name__ == "__main__":
    main()
