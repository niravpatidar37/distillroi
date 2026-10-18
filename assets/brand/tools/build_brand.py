# /// script
# requires-python = ">=3.12"
# dependencies = ["fonttools==4.66.1"]
# ///
"""Build every distillroi brand asset from source, reproducibly.

    uv run assets/brand/tools/build_brand.py

Requires `resvg` on PATH for the PNG exports (`cargo install resvg --locked --version 0.48.1`).

The wordmark is set in IBM Plex Mono Bold and converted to outlines, so the logos render
identically without the font installed. The font is fetched from the pinned upstream release
and verified by SHA-256 (it is not vendored). IBM Plex is licensed under the SIL Open Font
License 1.1, which permits using its outlines in artwork such as a logo.

The mark is a funnel with a single drop below it: a large model's behaviour, distilled into a
small one. The drop is the accent colour (copper, as in a copper still).
"""

from __future__ import annotations

import hashlib
import io
import pathlib
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

OUT = pathlib.Path(__file__).resolve().parents[1]
FONT_URL = "https://github.com/IBM/plex/releases/download/%40ibm%2Fplex-mono%401.1.0/ibm-plex-mono.zip"
FONT_SHA256 = "4bfc936d0e1fd19db6327a3786eabdbc3dc0d464500576f6458f6706df68d26c"
FONT_MEMBER = "ibm-plex-mono/fonts/complete/ttf/IBMPlexMono-{w}.ttf"

# Palette. One accent (copper), two tones so it passes contrast on both GitHub themes.
INK = "#14110F"  # mark on light backgrounds, tiles
SNOW = "#F5F0E8"  # mark on dark backgrounds
COPPER_ON_LIGHT = "#9C4A1A"
COPPER_ON_DARK = "#F0975A"
MUTED_ON_DARK = "#A8A29E"
GITHUB_LIGHT_BG = "#FFFFFF"
GITHUB_DARK_BG = "#0D1117"


# ----------------------------------------------------------------- contrast


def _lum(hex_color: str) -> float:
    rgb = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# ----------------------------------------------------------------- fonts


def load_fonts() -> dict[str, TTFont]:
    cache = OUT / "tools" / ".cache" / "ibm-plex-mono-1.1.0.zip"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FONT_URL, timeout=60) as r:
            cache.write_bytes(r.read())
    data = cache.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != FONT_SHA256:
        cache.unlink()
        sys.exit(f"font archive checksum mismatch: {digest}")
    z = zipfile.ZipFile(io.BytesIO(data))
    return {w: TTFont(io.BytesIO(z.read(FONT_MEMBER.format(w=w)))) for w in ("Bold", "Medium")}


def _ntos(n: float) -> str:
    s = f"{n:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def text_path(
    font: TTFont, text: str, size: float, x: float, baseline: float, tracking: float = 0.0
) -> tuple[str, float]:
    """Outline `text` as an SVG path. Returns (d, advance width)."""
    glyphs, cmap = font.getGlyphSet(), font.getBestCmap()
    scale = size / font["head"].unitsPerEm
    pen = SVGPathPen(glyphs, ntos=_ntos)
    cursor = x
    for ch in text:
        name = cmap[ord(ch)]
        glyphs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, cursor, baseline)))
        cursor += font["hmtx"][name][0] * scale + tracking
    return pen.getCommands(), cursor - x - tracking


def x_height(font: TTFont, size: float) -> float:
    return font["OS/2"].sxHeight * size / font["head"].unitsPerEm


# ----------------------------------------------------------------- the mark

FUNNEL = "M9 9H55L37 31V38H27V31Z"
DROP = "M32 43C32 43 25.5 50.5 25.5 54.5A6.5 6.5 0 0 0 38.5 54.5C38.5 50.5 32 43 32 43Z"
LEVEL = "M22.5 16H41.5"


def mark(fg: str, accent: str, *, x: float = 0, y: float = 0, size: float = 64) -> str:
    """Funnel + drop on a 64-unit grid. The short line inside the funnel is the liquid level."""
    s = size / 64
    return (
        f'<g transform="translate({_ntos(x)} {_ntos(y)}) scale({_ntos(s)})">'
        f'<path d="{FUNNEL}" fill="none" stroke="{fg}" stroke-width="5" stroke-linejoin="round"/>'
        f'<path d="{LEVEL}" fill="none" stroke="{accent}" stroke-width="3.5" stroke-linecap="round"/>'
        f'<path d="{DROP}" fill="{accent}"/>'
        "</g>"
    )


def tile(size: float, rx: float) -> str:
    """The mark on its own dark tile (favicon, avatar). rx=0 gives an opaque square."""
    inset = size * 0.14
    inner = size - 2 * inset
    return f'<rect width="{_ntos(size)}" height="{_ntos(size)}" rx="{_ntos(rx)}" fill="{INK}"/>' + mark(
        SNOW, COPPER_ON_DARK, x=inset, y=inset, size=inner
    )


def favicon16() -> str:
    """Hand-hinted 16px variant: strokes on the pixel grid so the funnel stays a funnel."""
    return (
        f'<rect width="16" height="16" rx="3" fill="{INK}"/>'
        '<g fill="none" stroke-width="1" shape-rendering="crispEdges">'
        f'<path stroke="{SNOW}" d="M2 2.5H14M2.5 2.5L6.5 7.5V10.5H9.5V7.5L13.5 2.5"/>'
        "</g>"
        f'<rect x="7" y="12" width="2" height="2" fill="{COPPER_ON_DARK}" shape-rendering="crispEdges"/>'
    )


def svg(width: float, height: float, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_ntos(width)}" height="{_ntos(height)}" '
        f'viewBox="0 0 {_ntos(width)} {_ntos(height)}" role="img" aria-label="{title}">'
        f"<title>{title}</title>{body}</svg>\n"
    )


def lockup(fonts: dict[str, TTFont], fg: str, accent: str) -> str:
    size, gap, tr = 44.0, 14.0, -0.8
    xh = x_height(fonts["Bold"], size)
    baseline = 32 + xh / 2
    d1, w1 = text_path(fonts["Bold"], "distill", size, 64 + gap, baseline, tracking=tr)
    d2, w2 = text_path(fonts["Bold"], "roi", size, 64 + gap + w1 + tr, baseline, tracking=tr)
    width = 64 + gap + w1 + tr + w2 + 2
    body = mark(fg, accent) + f'<path fill="{fg}" d="{d1}"/><path fill="{accent}" d="{d2}"/>'
    return svg(width, 64, body, "distillroi")


def social(fonts: dict[str, TTFont]) -> str:
    W, H, left = 1280, 640, 96
    b, md = fonts["Bold"], fonts["Medium"]
    parts = [f'<rect width="{W}" height="{H}" fill="{INK}"/>']
    parts.append(mark(SNOW, COPPER_ON_DARK, x=left - 8, y=64, size=120))
    wsize = 88
    base = 64 + 60 + x_height(b, wsize) / 2
    d1, w1 = text_path(b, "distill", wsize, left + 136, base, tracking=-1.5)
    d2, _ = text_path(b, "roi", wsize, left + 136 + w1 - 1.5, base, tracking=-1.5)
    parts.append(f'<path fill="{SNOW}" d="{d1}"/><path fill="{COPPER_ON_DARK}" d="{d2}"/>')
    for i, line in enumerate(
        ("Distill a frontier-model task", "into a small open model,", "and prove the unit economics.")
    ):
        d, _ = text_path(md, line, 46, left, 290 + i * 60)
        parts.append(f'<path fill="{SNOW}" d="{d}"/>')
    for i, line in enumerate(
        (
            "1.5B student on vLLM: 90% of teacher accuracy alone,",
            "99.7% with a confidence router; break-even ~313K req/day",
        )
    ):
        d, _ = text_path(md, line, 25, left, 476 + i * 36)
        parts.append(f'<path fill="{MUTED_ON_DARK}" d="{d}"/>')
    d, _ = text_path(md, "github.com/niravpatidar37/distillroi", 24, left, 586)
    parts.append(f'<path fill="{COPPER_ON_DARK}" d="{d}"/>')
    return svg(W, H, "".join(parts), "distillroi: distill a frontier-model task into a small open model")


# ----------------------------------------------------------------- safety gate

FORBIDDEN = [
    (re.compile(r"<\s*script", re.I), "script element"),
    (re.compile(r"\son[a-z]+\s*=", re.I), "event handler attribute"),
    (re.compile(r"<\s*foreignObject", re.I), "foreignObject"),
    (re.compile(r"(?:xlink:)?href\s*=\s*\"(?!#)", re.I), "external reference"),
    (re.compile(r"url\(\s*['\"]?(?!#)", re.I), "external url()"),
    (re.compile(r"javascript:", re.I), "javascript: URL"),
]


def check_safe(name: str, text: str) -> None:
    for pattern, what in FORBIDDEN:
        if pattern.search(text):
            sys.exit(f"{name}: unsafe SVG content ({what})")


# ----------------------------------------------------------------- main


def main() -> None:
    checks = {
        "accent on GitHub light": contrast(COPPER_ON_LIGHT, GITHUB_LIGHT_BG),
        "accent on GitHub dark": contrast(COPPER_ON_DARK, GITHUB_DARK_BG),
        "ink on GitHub light": contrast(INK, GITHUB_LIGHT_BG),
        "snow on GitHub dark": contrast(SNOW, GITHUB_DARK_BG),
        "favicon drop on tile": contrast(COPPER_ON_DARK, INK),
        "muted text on tile": contrast(MUTED_ON_DARK, INK),
    }
    for label, ratio in checks.items():
        print(f"contrast {label}: {ratio:.2f}:1")
        if ratio < 3.0:
            sys.exit(f"contrast below WCAG 3:1 for graphics: {label}")

    fonts = load_fonts()
    files = {
        "icon.svg": svg(64, 64, mark(INK, COPPER_ON_LIGHT), "distillroi"),
        "icon-dark.svg": svg(64, 64, mark(SNOW, COPPER_ON_DARK), "distillroi"),
        "favicon.svg": svg(32, 32, tile(32, 7), "distillroi"),
        "favicon-16.svg": svg(16, 16, favicon16(), "distillroi"),
        "avatar.svg": svg(512, 512, tile(512, 0), "distillroi"),
        "logo.svg": lockup(fonts, INK, COPPER_ON_LIGHT),
        "logo-dark.svg": lockup(fonts, SNOW, COPPER_ON_DARK),
        "social-preview.svg": social(fonts),
    }
    for name, text in files.items():
        check_safe(name, text)
        (OUT / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {name} ({len(text)} bytes)")

    resvg = shutil.which("resvg")
    if not resvg:
        sys.exit("resvg not found on PATH; install it to export PNGs")
    renders = [
        ("favicon-16.svg", "favicon-16.png", 16),
        ("favicon.svg", "favicon-32.png", 32),
        ("avatar.svg", "avatar-512.png", 512),
        ("social-preview.svg", "social-preview.png", 1280),
    ]
    for src, dst, width in renders:
        # List-form argv, no shell. All arguments are constants from `renders` plus the absolute
        # resvg path from shutil.which, so no untrusted input reaches this call.
        argv = [resvg, "-w", str(width), str(OUT / src), str(OUT / dst)]
        subprocess.run(argv, check=True, shell=False)  # noqa: S603  # nosec B603
        print(f"rendered {dst} ({(OUT / dst).stat().st_size} bytes)")


if __name__ == "__main__":
    main()
