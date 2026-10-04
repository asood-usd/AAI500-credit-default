"""Build the final report PDF: report body + executed notebooks as the appendix.

Usage (from the repo root, after running notebooks 01-04):
    python reports/build_report.py [output.pdf]

Steps:
1. reports/final_report.md -> HTML (mistune) -> PDF (headless Google Chrome)
2. each notebook -> HTML (nbconvert) -> PDF (headless Chrome)
3. merge everything into one PDF (pypdf)

Needs Google Chrome installed. On Windows/Linux, set CHROME to the browser path.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import mistune
from pypdf import PdfWriter

REPORTS = Path(__file__).resolve().parent
ROOT = REPORTS.parent
BUILD = REPORTS / "_build"
NOTEBOOKS = ["01_data_prep_eda", "02_eda_statistical_tests", "03_modeling", "04_evaluation"]
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

REPORT_CSS = """
@page { size: Letter; margin: 0.9in 0.9in; }
body { font-family: "Times New Roman", Times, serif; font-size: 11.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 22pt; line-height: 1.25; margin: 0 0 24px; }
h2 { font-size: 15pt; margin-top: 26px; border-bottom: 1px solid #999; padding-bottom: 3px; break-after: avoid; }
h3 { font-size: 12.5pt; margin-top: 18px; break-after: avoid; }
p, li { text-align: left; }
img { display: block; max-width: 92%; max-height: 7.2in; margin: 10px auto 4px; break-inside: avoid; }
p:has(> img) { break-inside: avoid; break-after: avoid; }
table { border-collapse: collapse; margin: 8px 0 14px; font-size: 10pt; width: 100%; break-inside: avoid; }
th, td { border: 1px solid #aaa; padding: 4px 6px; vertical-align: top; }
th { background: #eee; }
code { font-family: Menlo, Consolas, monospace; font-size: 9.5pt; }
.keep-together { break-inside: avoid; }
.title-page { text-align: center; padding-top: 2.2in; break-after: page; }
.title-page h1 { font-size: 24pt; }
.title-page p { text-align: center; font-size: 13pt; line-height: 1.7; }
"""

NOTEBOOK_CSS = """
<style>
@page { size: Letter; margin: 0.6in; }
pre { white-space: pre-wrap !important; word-break: break-word; }
.jp-OutputArea-output img, .output_png img { break-inside: avoid; max-width: 100%; }
.jp-Cell { break-inside: avoid-page; }
h1, h2, h3 { break-after: avoid; }
.jp-MarkdownCell:has(h2:last-child), .jp-MarkdownCell:has(h3:last-child) { break-after: avoid; }
</style>
"""


def print_pdf(html_path, pdf_path):
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--run-all-compositor-stages-before-draw", "--virtual-time-budget=20000",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                   check=True, capture_output=True)


def build_report_body():
    md = (REPORTS / "final_report.md").read_text()
    render = mistune.create_markdown(escape=False, plugins=["table"])
    # Markdown inside a raw <div> isn't rendered, so render the title page on its own.
    title_md, rest_md = md.split("</div>", 1)
    title_md = title_md.replace('<div class="title-page">', "")
    body = f'<div class="title-page">{render(title_md)}</div>{render(rest_md)}'
    # Keep each "Table N." caption on the same page as its table.
    body = re.sub(r"(<p><strong>Table \d+\.</strong>.*?</p>\s*<table>.*?</table>)",
                  r'<div class="keep-together">\1</div>', body, flags=re.S)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<base href="{REPORTS.as_uri()}/"><title>Final Project Report</title>
<style>{REPORT_CSS}</style></head><body>{body}</body></html>"""
    out = BUILD / "report.html"
    out.write_text(html)
    print_pdf(out, BUILD / "report.pdf")
    return BUILD / "report.pdf"


def build_notebook(name):
    nb_path = ROOT / "notebooks" / f"{name}.ipynb"
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", str(nb_path),
                    "--output-dir", str(BUILD)], check=True, capture_output=True)
    html_path = BUILD / f"{name}.html"
    html = html_path.read_text().replace("</head>", NOTEBOOK_CSS + "</head>", 1)
    html_path.write_text(html)
    print_pdf(html_path, BUILD / f"{name}.pdf")
    return BUILD / f"{name}.pdf"


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORTS / "Final-Project-Report-Team-02.pdf"
    BUILD.mkdir(exist_ok=True)
    parts = [build_report_body()] + [build_notebook(n) for n in NOTEBOOKS]
    writer = PdfWriter()
    for part in parts:
        writer.append(str(part))
    writer.write(str(output))
    print("Wrote", output, f"({len(writer.pages)} pages)")


if __name__ == "__main__":
    main()
