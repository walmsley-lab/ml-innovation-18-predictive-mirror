"""Tiny dependency-free renderers for the offline predictive-mirror lab.

Nothing here runs a model. RGB pixel rasters and structured 2D coordinates
are rendered separately, so distinct representations stay distinguishable.
"""
import base64
import html
import json
import struct
import zlib

import numpy as np


def png_bytes(rgb):
    """Encode [H,W,3] float [0,1] as PNG using only Python's stdlib."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image [H,W,3]")
    pixels = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    height, width = pixels.shape[:2]
    def chunk(name, data):
        return (struct.pack(">I", len(data)) + name + data +
                struct.pack(">I", zlib.crc32(name + data) & 0xffffffff))
    scanlines = b"".join(b"\0" + pixels[row].tobytes()
                         for row in range(height))
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(scanlines, level=6)) +
            chunk(b"IEND", b""))


def raster_points(points, edges=(), size=160, visible=None):
    """Draw supplied normalized 2D positions, never infer them from RGB."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2 or size < 16:
        raise ValueError("Expected [nodes,2] normalized locations")
    background = np.asarray((.075, .11, .15), np.float32)
    image = np.broadcast_to(background, (size, size, 3)).copy()
    visibility = (np.ones(len(points), bool) if visible is None
                  else np.asarray(visible, bool).reshape(-1))
    if len(visibility) != len(points):
        raise ValueError("Visibility must have one flag per point")
    pixel = np.rint(np.clip(points, 0, 1) * (size-1)).astype(int)
    # Supplied edges are a known prior (skeleton), not learned from images.
    for start, end in edges:
        if visibility[start] and visibility[end]:
            for alpha in np.linspace(0, 1, 75):
                xy = np.rint((1-alpha)*pixel[start]+alpha*pixel[end]).astype(int)
                x, y = xy
                image[max(0, y-1):min(size, y+2),
                      max(0, x-1):min(size, x+2)] = (.32, .48, .57)
    colors = ((.93, .43, .34), (.39, .81, .77), (.97, .76, .30),
              (.65, .54, .93))
    yy, xx = np.ogrid[:size, :size]
    for idx, (x, y) in enumerate(pixel):
        if visibility[idx]:
            circle = (xx-x)**2 + (yy-y)**2 <= 3**2
            image[circle] = colors[idx % len(colors)]
    return image


def html_document(items, title="Predictive Mirror Laboratory"):
    """Single-file offline viewer with a selector; no server or CDN."""
    if not items:
        raise ValueError("At least one model is required")
    tabs, panels = [], []
    for i, item in enumerate(items):
        ident = f"model-{i}"
        tabs.append('<option value="' + ident + '">' +
                    html.escape(item["label"]) + '</option>')
        cells = []
        for row in item["rows"]:
            tiles = []
            for heading, rgb in row["tiles"]:
                encoded = base64.b64encode(png_bytes(rgb)).decode("ascii")
                tiles.append(
                    '<figure><figcaption>' + html.escape(heading) +
                    '</figcaption><img alt="' + html.escape(heading) +
                    '" src="data:image/png;base64,' + encoded + '"></figure>')
            cells.append('<section class="lead"><h3>+' +
                         str(int(row["lead"])) + ' frames</h3><div class="tiles">' +
                         "".join(tiles) + '</div><p class="score">' +
                         html.escape(row["score"]) + '</p></section>')
        metadata = item["representation"] + " · " + item["checkpoint_sha256"][:12]
        panels.append(
            '<article id="' + ident + '" class="model"' +
            (' ' if i == 0 else ' hidden ') + '><h2>' +
            html.escape(item["label"]) + '</h2><p>' +
            html.escape(metadata) + '</p><p>' +
            html.escape(item["caveat"]) + '</p>' +
            "".join(cells) + '</article>')
    # All experiment data is embedded at creation time; JS only switches panes.
    script = (
        'document.getElementById("picker").addEventListener("change", '
        'function(e){document.querySelectorAll(".model").forEach(function(x)'
        '{x.hidden=(x.id!==e.target.value);});});')
    style = (
        "body{font:15px system-ui,sans-serif;background:#111b27;color:#e8edf4;"
        "margin:0 auto;padding:20px;max-width:1280px}header{padding:10px 0 20px}"
        "h1{font-size:24px;margin:0 0 10px}h2{font-size:21px}h3{font-size:15px}"
        "p{color:#b8c7d8;line-height:1.5}select{padding:10px;border-radius:6px;"
        "background:#28394b;color:white;max-width:100%}.lead{border-top:1px solid "
        "#43576c;padding:12px 0}.tiles{display:flex;flex-wrap:wrap;gap:12px}"
        "figure{margin:0;min-width:148px;max-width:190px;flex:1}figcaption{"
        "min-height:28px;font-size:12px;color:#c7d5e3}img{width:100%;"
        "aspect-ratio:1;image-rendering:pixelated;background:#1c2c3b;"
        "border:1px solid #43576c;border-radius:5px}.score{font-size:12px}"
        "[hidden]{display:none!important}")
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>" + html.escape(title) + "</title><style>" + style +
        "</style></head><body><header><h1>" + html.escape(title) +
        "</h1><p>Saved checkpoints only · forecasts use past observations · "
        "future truth is displayed for evaluation. Structured coordinates and "
        "RGB pixels are separate tasks, not one ranking.</p><label for=\"picker\">"
        "Select a technique: </label><select id=\"picker\">" +
        "".join(tabs) + "</select></header>" +
        "".join(panels) + "<script>" + script + "</script></body></html>")
