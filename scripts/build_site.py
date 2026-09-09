"""
Build the static site published to Netlify.

`docs/explainer.html` is written as a *fragment*: it starts at <title> and has
no <html>/<head>/<body>, because the Artifact host supplies that wrapper. A
static host does not, so this script wraps the same source into a complete,
standalone document under `site/`.

Keeping one source file and wrapping it at build time means the explainer never
drifts between the Artifact version and the public site.

    python scripts/build_site.py          # writes site/index.html
"""

import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "explainer.html"
OUT_DIR = ROOT / "site"
OUT_FILE = OUT_DIR / "index.html"

# Matches the Artifact host's own reset so the standalone page looks identical.
HEAD_EXTRAS = """\
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="How DriftGuard caught a customer-support AI agent drifting past its authority and automatically revoked its permissions.">
<meta name="color-scheme" content="light dark">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>&#128737;</text></svg>">
<style>
  /* The Artifact host applies these; reproduce them so both render alike. */
  *, *::before, *::after { box-sizing: border-box; }
  img { max-width: 100%; }
  [hidden] { display: none !important; }
</style>"""


def build() -> int:
    if not SOURCE.exists():
        print(f"error: {SOURCE} not found", file=sys.stderr)
        return 1

    fragment = SOURCE.read_text(encoding="utf-8")

    # Pull the <title> out of the fragment so it lands in <head> properly.
    match = re.search(r"<title>(.*?)</title>", fragment, re.DOTALL)
    title = match.group(1).strip() if match else "DriftGuard"
    body = re.sub(r"<title>.*?</title>\s*", "", fragment, count=1, flags=re.DOTALL)

    document = (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        f"{HEAD_EXTRAS}\n"
        f"<title>{title}</title>\n"
        "</head>\n"
        "<body>\n"
        f"{body.strip()}\n"
        "</body>\n"
        "</html>\n"
    )

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    OUT_FILE.write_text(document, encoding="utf-8")

    print(f"built {OUT_FILE.relative_to(ROOT)}  ({len(document):,} bytes)")
    print(f"title: {title}")
    return 0


if __name__ == "__main__":
    sys.exit(build())
