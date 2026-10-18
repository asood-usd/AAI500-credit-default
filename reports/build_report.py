"""Build the final report PDF in APA 7 student-paper format, with the executed notebooks as appendices.

Usage (from the repo root, after running notebooks 01-04):
    python reports/build_report.py [output.pdf]

Steps:
1. reports/final_report.md -> HTML (mistune) -> PDF (headless Google Chrome)
2. each notebook -> HTML (nbconvert), labeled Appendix A-D -> PDF (headless Chrome)
3. merge everything into one PDF and stamp page numbers top right on every page (pypdf)

APA 7 formatting applied: Times New Roman 12 pt, double spacing, 1-inch margins, 0.5-inch paragraph
indents, page numbers top right, centered bold Level 1 and flush-left bold Level 2 headings, tables
and figures labeled in bold with italic titles, and references on a new page with hanging indents.

Needs Google Chrome installed. On Windows/Linux, set CHROME to the browser path.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import mistune
from pypdf import PdfReader, PdfWriter

REPORTS = Path(__file__).resolve().parent
ROOT = REPORTS.parent
BUILD = REPORTS / "_build"
NOTEBOOKS = [
    ("01_data_prep_eda", "A", "Notebook 01: Data Preparation and First Look"),
    ("02_eda_statistical_tests", "B", "Notebook 02: EDA and Hypothesis Tests"),
    ("03_modeling", "C", "Notebook 03: Logistic Regression and Comparison Models"),
    ("04_evaluation", "D", "Notebook 04: Evaluation, Choosing a Cutoff, and What It Means"),
]
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

REPORT_CSS = """
@page { size: Letter; margin: 1in; }
body { font-family: "Times New Roman", Times, serif; font-size: 12pt; line-height: 2; color: #000; }
p, li { text-align: left; margin: 0; }
body > p { text-indent: 0.5in; }
ul, ol { margin: 0; padding-left: 0.5in; }
h2, h3 { font-size: 12pt; font-weight: bold; line-height: 2; margin: 0; break-after: avoid; }
h2 { text-align: center; }                 /* APA Level 1 */
h3 { text-align: left; }                   /* APA Level 2 */
.paper-title { text-align: center; font-weight: bold; text-indent: 0 !important; }
code { font-family: "Courier New", Courier, monospace; font-size: 11pt; }
a { color: #000; text-decoration: none; }

/* title page */
.title-page { text-align: center; padding-top: 1.5in; break-after: page; }
.title-page p { text-align: center; text-indent: 0; }
.title-page .tp-title { font-weight: bold; }

/* tables and figures: bold number, italic title, note below; kept on one page */
.keep-together { break-inside: avoid; margin: 0 0 12pt; }
.keep-together p { text-indent: 0; }
table { border-collapse: collapse; width: 100%; font-size: 11pt; line-height: 1.3; margin: 4pt 0; }
th, td { border: none; padding: 4pt 6pt; vertical-align: top; text-align: left; }
thead th { border-top: 1px solid #000; border-bottom: 1px solid #000; font-weight: normal; }
tbody tr:last-child td { border-bottom: 1px solid #000; }
img { display: block; max-width: 100%; max-height: 6.2in; margin: 4pt auto; }

/* references: new page, hanging indent */
.references { break-before: page; }
.references p { padding-left: 0.5in; text-indent: -0.5in; }
"""

NOTEBOOK_CSS = """
<style>
@page { size: Letter; margin: 1in; }
pre { white-space: pre-wrap !important; word-break: break-word; }
.jp-OutputArea-output pre { font-size: 9.5px; line-height: 1.35; }
.jp-OutputArea-output img, .output_png img { break-inside: avoid; max-width: 100%; }
.jp-RenderedHTMLCommon table { font-size: 10px; }
h1, h2, h3, h4 { break-after: avoid; }
.apa-appendix { font-family: "Times New Roman", Times, serif; font-size: 12pt; font-weight: bold;
                text-align: center; line-height: 2; margin-bottom: 12pt; }
</style>
"""

PAGE_NUMBER_CSS = """
@page { size: Letter; margin: 0; }
body { margin: 0; }
.pg { position: relative; height: 11in; width: 8.5in; break-after: page; }
.pg span { position: absolute; top: 0.5in; right: 1in; font: 12pt "Times New Roman", Times, serif; }
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

    # Keep each table (label, title, table, note) and each figure (label, title, image, note) together.
    note = r"(?:\s*<p><em>Note\.</em>.*?</p>)?"
    body = re.sub(r"(<p><strong>Table \d+</strong></p>\s*<p><em>.*?</em></p>\s*<table>.*?</table>" + note + ")",
                  r'<div class="keep-together">\1</div>', body, flags=re.S)
    body = re.sub(r"(<p><strong>Figure \d+</strong></p>\s*<p><em>.*?</em></p>\s*<p><img.*?</p>" + note + ")",
                  r'<div class="keep-together">\1</div>', body, flags=re.S)
    # References start on a new page and use hanging indents.
    body = body.replace("<h2>References</h2>", '<div class="references"><h2>References</h2>', 1) + "</div>"

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<base href="{REPORTS.as_uri()}/"><title>Final Project Report</title>
<style>{REPORT_CSS}</style></head><body>{body}</body></html>"""
    out = BUILD / "report.html"
    out.write_text(html)
    print_pdf(out, BUILD / "report.pdf")
    return BUILD / "report.pdf"


def build_notebook(name, letter, title):
    nb_path = ROOT / "notebooks" / f"{name}.ipynb"
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", str(nb_path),
                    "--output-dir", str(BUILD)], check=True, capture_output=True)
    html_path = BUILD / f"{name}.html"
    html = html_path.read_text().replace("</head>", NOTEBOOK_CSS + "</head>", 1)
    label = f'<div class="apa-appendix">Appendix {letter}<br>{title}</div>'
    html = re.sub(r"(<body[^>]*>)", r"\1" + label, html, count=1)
    html_path.write_text(html)
    print_pdf(html_path, BUILD / f"{name}.pdf")
    return BUILD / f"{name}.pdf"


def page_numbers(n):
    """A PDF with only the page numbers, top right, to stamp onto every page."""
    pages = "".join(f'<div class="pg"><span>{i}</span></div>' for i in range(1, n + 1))
    html = BUILD / "page_numbers.html"
    html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{PAGE_NUMBER_CSS}</style></head>"
                    f"<body>{pages}</body></html>")
    print_pdf(html, BUILD / "page_numbers.pdf")
    return PdfReader(str(BUILD / "page_numbers.pdf"))


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORTS / "Final-Project-Report-Team-02.pdf"
    BUILD.mkdir(exist_ok=True)
    parts = [build_report_body()] + [build_notebook(*nb) for nb in NOTEBOOKS]
    writer = PdfWriter()
    for part in parts:
        writer.append(str(part))
    numbers = page_numbers(len(writer.pages))
    for page, number in zip(writer.pages, numbers.pages):
        page.merge_page(number)
    writer.write(str(output))
    print("Wrote", output, f"({len(writer.pages)} pages)")


if __name__ == "__main__":
    main()
