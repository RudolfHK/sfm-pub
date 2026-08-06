"""Baut aus paper_t.md eine druckfertige HTML-Datei.

Der Export nach PDF erfolgt anschließend im Browser über "Drucken" und
"Als PDF speichern" (A4, Ränder "Standard", Hintergrundgrafiken aktivieren).
Ein LaTeX- oder pandoc-Toolchain wird nicht benötigt.

    python paper/scripts/build_paper.py                 # verlinkte Bilder
    python paper/scripts/build_paper.py --embed         # Bilder eingebettet
    python paper/scripts/build_paper.py --md paper/paper_draft_v1.md

Abhängigkeit: markdown (pip install markdown)
"""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import markdown  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    sys.exit("Bitte zuerst installieren:  pip install markdown")

PAPER_DIR = Path(__file__).resolve().parents[1]

# Chromium-basierte Browser koennen headless direkt nach PDF drucken.
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

CSS = """
/* Layout nach abstract/workshop_book_styleguide_2026/main.tex
   A4, Rand 2,5 cm, Segoe UI, Fliesstext 9 pt / 14,4 pt, Ueberschriften
   zentriert in GFaI-Blau #23355D, keine Seitenzahlen. */

@page { size: A4; margin: 25mm; }

:root { --ink:#000000; --blue:#23355D; --rule:#000000; }

* { box-sizing: border-box; }

body {
  font-family: "Segoe UI", "Segoe UI Web", Frutiger, "Helvetica Neue", Arial, sans-serif;
  font-size: 9pt; line-height: 14.4pt; color: var(--ink);
  margin: 0 auto; padding: 25mm 0 40mm; max-width: 160mm; background: #fff;
  text-align: justify; hyphens: auto;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}

/* ---------- Titelkopf -------------------------------------------------- */
h1 {
  font-size: 12pt; line-height: 14pt; font-weight: 700; color: #000;
  text-align: center; margin: 0 0 9.5pt; hyphens: none;
}
p.authors {
  font-size: 10pt; line-height: 14.4pt; text-align: center; margin: 0 0 2pt;
}
p.affiliation {
  font-size: 10pt; line-height: 14.4pt; text-align: center; margin: 0 0 9pt;
}
p.abstract { margin: 9pt 0 0; }
p.abstract > strong:first-child,
p.keywords > strong:first-child {
  font-size: 10pt; font-weight: 700; color: var(--blue);
}
p.keywords { text-align: center; margin: 15pt 0 0; }

/* ---------- Ueberschriften --------------------------------------------- */
h2, h3 {
  font-size: 10pt; line-height: 12pt; font-weight: 700; color: var(--blue);
  text-align: center; hyphens: none;
  break-after: avoid; page-break-after: avoid;
}
h2 { margin: 22pt 0 8pt; }
h3 { margin: 14pt 0 6pt; }

/* ---------- Fliesstext -------------------------------------------------- */
p { margin: 0 0 7pt; }
hr { border: 0; border-top: .5pt solid #999; margin: 12pt 0; }
a { color: var(--blue); text-decoration: none; }
strong { font-weight: 700; }

code, kbd {
  font-family: Consolas, "Courier New", monospace; font-size: 8.5pt;
  background: #f2f3f5; padding: 0 2px; border-radius: 2px;
}

ul, ol { margin: 0 0 7pt; padding-left: 14pt; }
li { margin-bottom: 2pt; }

/* ---------- Abbildungen ------------------------------------------------- */
figure {
  margin: 12pt 0; text-align: center;
  break-inside: avoid; page-break-inside: avoid;
}
figure img { max-width: 100%; max-height: 112mm; width: auto; height: auto; }
figcaption {
  font-size: 9pt; line-height: 12pt; text-align: justify;
  margin-top: 3pt; color: var(--ink);
}
figcaption strong:first-child { color: var(--ink); }

/* Bildraster (die vier Rekonstruktionsschritte) */
figure.grid table { width: 100%; border-collapse: collapse; }
figure.grid table td, figure.grid table th {
  border: 0; padding: 2pt 4pt; vertical-align: middle;
  text-align: center; font-size: 8.5pt;
}
figure.grid table tbody tr { background: transparent; }
figure.grid table img { max-width: 100%; max-height: 76mm; }
figure.tab { margin: 10pt 0; }

/* ---------- Tabellen im booktabs-Stil ----------------------------------- */
table {
  width: 100%; border-collapse: collapse; font-size: 8.5pt; line-height: 11pt;
  margin: 4pt 0 3pt; break-inside: avoid; page-break-inside: avoid;
}
th, td {
  border: 0; border-bottom: .4pt solid #b9bec6;
  padding: 3pt 5pt; text-align: left; vertical-align: top;
}
thead th {
  border-top: 1pt solid var(--ink); border-bottom: .6pt solid var(--ink);
  font-weight: 700; background: transparent;
}
tbody tr:last-child td { border-bottom: 1pt solid var(--ink); }
figure.grid table thead th, figure.grid table tbody tr:last-child td { border: 0; }

/* ---------- Bildschirm-Hinweis ------------------------------------------ */
.hint {
  font-size: 8.5pt; line-height: 12pt; text-align: left;
  background: #fff8e6; border: 1px solid #f0d9a0; border-radius: 4px;
  padding: 8px 12px; margin-bottom: 24px; color: #5c4a1a;
}

@media print {
  body { padding: 0; max-width: none; }
  .hint { display: none; }
  a { color: inherit; }
}
"""

HINT = (
    '<div class="hint"><strong>PDF-Export:</strong> Im Browser '
    "<em>Drucken</em> (Strg&nbsp;+&nbsp;P) &rarr; Ziel <em>Als PDF speichern</em>, "
    "Papierformat A4, Option <em>Hintergrundgrafiken</em> aktivieren. "
    "Dieser Hinweis erscheint im PDF nicht.</div>"
)

CAPTION = r"<p><strong>(?:Abb|Fig|Tab)\.\s"


def embed_images(html_doc: str, base: Path) -> str:
    """Ersetzt src-Pfade durch data-URIs, damit die Datei allein lauffähig ist."""

    def repl(m: re.Match) -> str:
        src = m.group(1)
        if src.startswith(("data:", "http://", "https://")):
            return m.group(0)
        path = (base / src).resolve()
        if not path.is_file():
            print(f"  WARNUNG: Bild fehlt: {src}")
            return m.group(0)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'src="data:{mime};base64,{data}"'

    return re.sub(r'src="([^"]+)"', repl, html_doc)


def group_figures(html_doc: str) -> tuple[str, int]:
    """Bündelt Bild bzw. Bildtabelle mit der zugehörigen Bildunterschrift."""
    count = 0

    def wrap(m: re.Match) -> str:
        nonlocal count
        count += 1
        body, caption = m.group(1), m.group(2)
        caption = re.sub(r"^<p>|</p>$", "", caption.strip())
        if body.startswith("<table"):
            cls = ' class="grid"' if "<img" in body else ' class="tab"'
        else:
            cls = ""
        return f"<figure{cls}>\n{body}\n<figcaption>{caption}</figcaption>\n</figure>"

    # Bild + Unterschrift
    html_doc = re.sub(
        r"(<p>(?:<img[^>]*>\s*)+</p>)\s*(" + CAPTION + r".*?</p>)",
        wrap, html_doc, flags=re.DOTALL,
    )
    # Tabelle + Unterschrift (Bildraster wie Datentabellen)
    html_doc = re.sub(
        r"(<table>(?:(?!</table>).)*?</table>)\s*(" + CAPTION + r".*?</p>)",
        wrap, html_doc, flags=re.DOTALL,
    )
    return html_doc, count


def check_images(html_doc: str, base: Path) -> list[str]:
    missing = []
    for src in re.findall(r'<img[^>]*src="([^"]+)"', html_doc):
        if src.startswith(("data:", "http")):
            continue
        if not (base / src).is_file():
            missing.append(src)
    return missing


def print_to_pdf(html_path: Path) -> Path | None:
    """Druckt die HTML-Datei headless nach PDF (Edge oder Chrome)."""
    browser = next((b for b in BROWSERS if Path(b).is_file()), None)
    if browser is None:
        print("  Kein Edge/Chrome gefunden. PDF bitte im Browser ueber "
              "Drucken -> Als PDF speichern erzeugen.")
        return None

    pdf_path = html_path.with_suffix(".pdf")
    profile = Path(tempfile.mkdtemp(prefix="paperpdf_"))
    cmd = [
        browser, "--headless=new", "--disable-gpu", "--no-sandbox",
        f"--user-data-dir={profile}", "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw", "--virtual-time-budget=20000",
        f"--print-to-pdf={pdf_path}", html_path.as_uri(),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=300)
    finally:
        shutil.rmtree(profile, ignore_errors=True)
    if pdf_path.is_file():
        print(f"geschrieben: {pdf_path}  ({pdf_path.stat().st_size / 1e6:.1f} MB)")
        return pdf_path
    print("  PDF-Export fehlgeschlagen, bitte im Browser drucken.")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(PAPER_DIR / "paper_t.md"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--embed", action="store_true",
                    help="Bilder als data-URI einbetten (eine einzelne Datei)")
    ap.add_argument("--pdf", action="store_true",
                    help="zusaetzlich headless nach PDF drucken (Edge/Chrome)")
    args = ap.parse_args()

    md_path = Path(args.md).resolve()
    out_path = Path(args.out) if args.out else md_path.with_suffix(".html")
    base = md_path.parent

    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        text, extensions=["tables", "attr_list", "sane_lists", "smarty"],
        extension_configs={"smarty": {"smart_dashes": False}},
    )

    body, n_fig = group_figures(body)

    missing = check_images(body, base)
    for src in missing:
        print(f"  WARNUNG: Bild fehlt: {src}")

    n_img = len(re.findall(r"<img", body))
    if args.embed:
        body = embed_images(body, base)

    title = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.DOTALL)
    title_txt = re.sub(r"<[^>]+>", "", title.group(1)) if title else md_path.stem

    doc = (
        "<!doctype html>\n<html lang=\"de\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title_txt)}</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n{HINT}\n{body}\n</body>\n</html>\n"
    )
    out_path.write_text(doc, encoding="utf-8")

    size_mb = out_path.stat().st_size / 1e6
    print(f"geschrieben: {out_path}  ({size_mb:.1f} MB)")
    print(f"  Abbildungen gebuendelt: {n_fig}")
    print(f"  Bilder eingebunden:     {n_img}")
    print(f"  fehlende Bilder:        {len(missing)}")

    if args.pdf:
        print_to_pdf(out_path)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
