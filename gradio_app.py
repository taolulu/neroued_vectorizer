"""Gradio Web UI for neroued-vectorizer service."""

from __future__ import annotations

import io
import re
import tempfile
from pathlib import Path

import cairosvg
import gradio as gr

import neroued_vectorizer as nv

# ── SVG Preview JavaScript ───────────────────────────────────────────────────

SVG_PREVIEW_JS = """
<script>
// ── Pan & Zoom state ──────────────────────────────────────────────────────
window._svgPan = { dragging: false, lastX: 0, lastY: 0 };

function _initSvgInteraction() {
    var outer = document.getElementById('svg-preview-outer');
    if (!outer) return;
    // Remove old listeners by cloning
    var newOuter = outer.cloneNode(true);
    outer.parentNode.replaceChild(newOuter, outer);

    newOuter.addEventListener('wheel', function(e) {
        e.preventDefault();
        var inner = document.getElementById('svg-preview-inner');
        if (!inner) return;
        var cur = inner.style.transform.match(/scale\\(([\\d.]+)\\)/);
        var level = cur ? parseFloat(cur[1]) : 1;
        var delta = e.deltaY < 0 ? 0.1 : -0.1;
        level = Math.max(0.1, Math.min(20, level + delta));
        inner.style.transform = 'scale(' + level + ')';
        inner.style.transformOrigin = (e.clientX - newOuter.getBoundingClientRect().left) + 'px ' + (e.clientY - newOuter.getBoundingClientRect().top) + 'px';
    }, { passive: false });

    newOuter.addEventListener('mousedown', function(e) {
        if (e.button !== 0) return;
        window._svgPan.dragging = true;
        window._svgPan.lastX = e.clientX;
        window._svgPan.lastY = e.clientY;
        newOuter.style.cursor = 'grabbing';
    });

    document.addEventListener('mousemove', function(e) {
        if (!window._svgPan.dragging) return;
        var inner = document.getElementById('svg-preview-inner');
        if (!inner) return;
        var dx = e.clientX - window._svgPan.lastX;
        var dy = e.clientY - window._svgPan.lastY;
        window._svgPan.lastX = e.clientX;
        window._svgPan.lastY = e.clientY;
        var cur = inner.style.transform.match(/translate\\(([\\d.-]+)px, ([\\d.-]+)px\\)\\s+scale\\(([\\d.]+)\\)/);
        var tx = 0, ty = 0, scale = 1;
        if (cur) { tx = parseFloat(cur[1]); ty = parseFloat(cur[2]); scale = parseFloat(cur[3]); }
        tx += dx; ty += dy;
        inner.style.transform = 'translate(' + tx + 'px, ' + ty + 'px) scale(' + scale + ')';
        inner.style.transformOrigin = 'center center';
    });

    document.addEventListener('mouseup', function() {
        if (!window._svgPan.dragging) return;
        window._svgPan.dragging = false;
        var outer2 = document.getElementById('svg-preview-outer');
        if (outer2) outer2.style.cursor = '';
    });
}

// ── Pan & Zoom ─────────────────────────────────────────────────────────────
function setSvgZoom(level) {
    var inner = document.getElementById('svg-preview-inner');
    if (inner) {
        var m = inner.style.transform.match(/translate\\(([\\d.-]+)px, ([\\d.-]+)px\\)/);
        var tx = m ? m[1] : '0';
        var ty = m ? m[2] : '0';
        inner.style.transform = 'translate(' + tx + 'px, ' + ty + 'px) scale(' + level + ')';
        inner.style.transformOrigin = 'center center';
    }
}

function setSvgBg(bg) {
    var outer = document.getElementById('svg-preview-outer');
    if (!outer) return;
    switch(bg) {
        case 'transparent':
            outer.style.background = 'repeating-conic-gradient(#e0e0e0 0% 25%, #fff 0% 50%) 50%/16px 16px';
            break;
        case 'white':
            outer.style.background = '#ffffff';
            break;
        case 'black':
            outer.style.background = '#111111';
            break;
        case 'checker':
        default:
            outer.style.background = 'repeating-conic-gradient(#ccc 0% 25%, #fff 0% 50%) 50%/16px 16px';
    }
}

function fitSvgToContainer() {
    var outer = document.getElementById('svg-preview-outer');
    var inner = document.getElementById('svg-preview-inner');
    if (!outer || !inner) return;
    var svg = inner.querySelector('svg');
    if (!svg) return;
    var vb = svg.getAttribute('viewBox');
    if (vb) {
        var parts = vb.split(' ').map(Number);
        var w = parts[2], h = parts[3];
    } else {
        var w = parseFloat(svg.getAttribute('width')) || inner.offsetWidth;
        var h = parseFloat(svg.getAttribute('height')) || inner.offsetHeight;
    }
    var scaleX = (outer.offsetWidth - 16) / w;
    var scaleY = (outer.offsetHeight - 16) / h;
    var scale = Math.min(scaleX, scaleY, 2);
    inner.style.transform = 'translate(0px,0px) scale(' + scale + ')';
    inner.style.transformOrigin = 'center center';
}

function actualSvgSize() {
    var inner = document.getElementById('svg-preview-inner');
    if (inner) {
        inner.style.transform = 'translate(0px,0px) scale(1)';
        inner.style.transformOrigin = 'center center';
    }
}

function zoomSvg(delta) {
    var inner = document.getElementById('svg-preview-inner');
    if (!inner) return;
    var cur = inner.style.transform.match(/scale\\(([\\d.]+)\\)/);
    var level = cur ? parseFloat(cur[1]) : 1;
    level = Math.max(0.1, Math.min(20, level + delta));
    var m = inner.style.transform.match(/translate\\(([\\d.-]+)px, ([\\d.-]+)px\\)/);
    var tx = m ? m[1] : '0';
    var ty = m ? m[2] : '0';
    inner.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + level + ')';
    inner.style.transformOrigin = 'center center';
}

// ── Fullscreen ─────────────────────────────────────────────────────────────
function fullscreenPreview() {
    var outer = document.getElementById('svg-preview-outer');
    if (!outer) return;
    if (!document.fullscreenElement) {
        outer.requestFullscreen().catch(function(e){ console.error(e); });
    } else {
        document.exitFullscreen();
    }
}

// ── Highlight paths in SVG by fill color ───────────────────────────────────
function highlightSvgColor(hexColor, active) {
    var inner = document.getElementById('svg-preview-inner');
    if (!inner) return;
    var paths = inner.querySelectorAll('path, rect, circle, ellipse, polygon, polyline');
    paths.forEach(function(p) {
        var fill = p.getAttribute('fill') || '';
        var normFill = fill.replace('#', '').toLowerCase();
        var normTarget = hexColor.replace('#', '').toLowerCase();
        if (normFill === normTarget) {
            if (active) {
                p.style.filter = 'drop-shadow(0 0 4px ' + hexColor + ')';
                p.style.strokeWidth = '2';
            } else {
                p.style.filter = '';
                p.style.strokeWidth = '';
            }
        }
    });
}

// Reset all SVG path highlighting
function resetSvgHighlight() {
    var inner = document.getElementById('svg-preview-inner');
    if (!inner) return;
    var paths = inner.querySelectorAll('path, rect, circle, ellipse, polygon, polyline');
    paths.forEach(function(p) {
        p.style.filter = '';
        p.style.strokeWidth = '';
    });
}

// ── Re-init interaction after SVG updates ──────────────────────────────────
window._svgInitDone = false;
function scheduleSvgInit() {
    if (window._svgInitDone) return;
    window._svgInitDone = true;
    setTimeout(function() {
        _initSvgInteraction();
        window._svgInitDone = false;
    }, 100);
}
</script>
"""

COPY_FEEDBACK_JS = """
<script>
async function copySvgNow() {
    var src = document.getElementById('svg-source-hidden');
    if (!src) return;
    try {
        await navigator.clipboard.writeText(src.value || '');
        var btn = document.getElementById('copy-svg-btn');
        if (btn) {
            var orig = btn.textContent;
            btn.textContent = 'Copied!';
            btn.style.color = '#22c55e';
            setTimeout(function() {
                btn.textContent = orig;
                btn.style.color = '';
            }, 1500);
        }
    } catch(e) {
        console.error('Copy failed:', e);
    }
}
</script>
"""

# ── SVG Helpers ───────────────────────────────────────────────────────────────

def parse_svg_color_stats(svg_content: str) -> dict[str, dict]:
    """Parse SVG and return per-color path count and percentage."""
    # Find all fill colors used in paths
    fill_counts: dict[str, int] = {}
    path_pattern = re.compile(
        r'<(?P<tag>path|rect|circle|ellipse|polygon|polyline)\b[^>]*\bfill="(?P<color>#[a-fA-F0-9]+)"',
        re.IGNORECASE
    )
    for m in path_pattern.finditer(svg_content):
        color = m.group('color').lower()
        fill_counts[color] = fill_counts.get(color, 0) + 1

    total = sum(fill_counts.values()) or 1
    return {
        color: {
            'count': count,
            'pct': round(count / total * 100, 1),
        }
        for color, count in fill_counts.items()
    }


def render_svg_to_png_bytes(svg_content: str, width: int, height: int) -> bytes:
    """Render SVG content to PNG bytes using cairosvg."""
    return cairosvg.svg2png(
        bytestring=svg_content.encode('utf-8'),
        output_width=width,
        output_height=height,
    )


def build_enhanced_palette_html(
    svg_content: str,
    palette: list,
) -> str:
    """Build interactive palette HTML with usage % and highlighting."""
    stats = parse_svg_color_stats(svg_content)
    swatches = []
    for i, color in enumerate(palette):
        r255 = int(round(color.r * 255))
        g255 = int(round(color.g * 255))
        b255 = int(round(color.b * 255))
        hex_color = f"#{r255:02x}{g255:02x}{b255:02x}"
        stat = stats.get(hex_color.lower(), {'count': 0, 'pct': 0.0})
        text_color = "#fff" if (r255 * 0.299 + g255 * 0.587 + b255 * 0.114) < 150 else "#000"
        swatches.append(
            f'<div class="palette-swatch" '
            f'data-color="{hex_color}" '
            f'onclick="window._togglePaletteHighlight(this, \'{hex_color}\')" '
            f'style="display:inline-flex;flex-direction:column;align-items:center;'
            f'margin:4px;cursor:pointer;" '
            f'title="{hex_color} — {stat["count"]} shapes ({stat["pct"]}%)">'
            f'<div style="width:48px;height:48px;background:{hex_color};'
            f'border:2px solid transparent;border-radius:6px;'
            f'box-shadow:0 1px 4px rgba(0,0,0,0.2);'
            f'transition:border-color 0.15s,transform 0.15s;" '
            f'onmouseover="this.style.borderColor=\'#3b82f6\';this.style.transform=\'scale(1.1)\'" '
            f'onmouseout="this.style.borderColor=\'transparent\';this.style.transform=\'scale(1)\'">'
            f'<span style="display:block;padding-top:14px;font-size:12px;'
            f'font-weight:bold;color:{text_color};font-family:monospace;">{i+1}</span></div>'
            f'<span style="font-size:10px;color:#666;margin-top:3px;font-family:monospace;">'
            f'{stat["pct"]}%</span></div>'
        )
    return (
        f'<div id="palette-container" style="display:flex;flex-wrap:wrap;gap:2px;padding:4px 0;">'
        + ''.join(swatches)
        + '</div>'
        + f'<div style="margin-top:8px;font-size:12px;color:#888;">'
        f'Click a swatch to highlight its regions in the SVG above</div>'
    )


def build_palette_export_css(palette: list) -> str:
    """Export palette as CSS variables."""
    lines = [":root {"]
    for i, color in enumerate(palette):
        r255 = int(round(color.r * 255))
        g255 = int(round(color.g * 255))
        b255 = int(round(color.b * 255))
        lines.append(f"  --color-{i+1}: #{r255:02x}{g255:02x}{b255:02x};")
    lines.append("}")
    return "\n".join(lines)


# ── Vectorization ─────────────────────────────────────────────────────────────

def vectorize_image(
    image_file: str | bytes,
    num_colors: int,
    min_region_area: int,
    curve_fit_error: float,
    corner_angle_threshold: float,
    smoothness: float,
    smoothing_spatial: float,
    smoothing_color: float,
    upscale_short_edge: int,
    max_working_pixels: int,
    enable_subpixel_refine: bool,
) -> tuple[str, str, str, str, bytes | None, str]:
    """Vectorize an image and return SVG content, metadata, palette HTML, palette CSS, PNG bytes, original image path."""
    config = nv.VectorizerConfig()
    config.num_colors = num_colors
    config.min_region_area = min_region_area
    config.curve_fit_error = curve_fit_error
    config.corner_angle_threshold = corner_angle_threshold
    config.smoothness = smoothness
    config.smoothing_spatial = smoothing_spatial
    config.smoothing_color = smoothing_color
    config.upscale_short_edge = upscale_short_edge
    config.max_working_pixels = max_working_pixels
    config.enable_subpixel_refine = enable_subpixel_refine

    result = nv.vectorize(image_file, config)

    palette_html = build_enhanced_palette_html(result.svg_content, result.palette)
    palette_css = build_palette_export_css(result.palette)

    metadata = (
        f"Width: {result.width} px\n"
        f"Height: {result.height} px\n"
        f"Shapes: {result.num_shapes}\n"
        f"Colors: {result.resolved_num_colors}\n"
        + "\n".join(
            f"  {i+1}. #{int(c.r*255):02x}{int(c.g*255):02x}{int(c.b*255):02x}"
            for i, c in enumerate(result.palette)
        )
    )

    # Render SVG to PNG for comparison slider
    png_bytes = None
    try:
        png_bytes = render_svg_to_png_bytes(result.svg_content, result.width, result.height)
    except Exception as e:
        print(f"[Warning] Could not render SVG to PNG: {e}")

    return (
        result.svg_content,
        metadata,
        palette_html,
        palette_css,
        png_bytes,
        image_file,
    )


# ── App constants ─────────────────────────────────────────────────────────────

BLOCK_TITLE = "neroued-vectorizer"
DESCRIPTION = (
    "High-quality raster-to-SVG vectorizer. Upload an image, tune the parameters "
    "below, and click **Vectorize** to generate the SVG."
)

SVG_PLACEHOLDER = (
    '<div id="svg-preview-outer" style="width:100%;height:460px;display:flex;'
    'align-items:center;justify-content:center;border:1px dashed #ccc;'
    'background:#ffffff;box-sizing:border-box;overflow:auto;cursor:default;">'
    '<div id="svg-preview-inner" style="color:#aaa;font-size:14px;text-align:center;padding:20px;">'
    'SVG preview will appear here after vectorization'
    '</div></div>'
)

SVG_LOADING = (
    '<div id="svg-preview-outer" style="width:100%;height:460px;display:flex;'
    'align-items:center;justify-content:center;'
    'background:#ffffff;box-sizing:border-box;overflow:hidden;position:relative;">'
    '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;">'
    '<div style="text-align:center;">'
    '<svg width="48" height="48" viewBox="0 0 48 48" style="animation:spin 1.5s linear infinite;">'
    '<circle cx="24" cy="24" r="20" fill="none" stroke="#e0e0e0" stroke-width="4"/>'
    '<path d="M24 4 A20 20 0 0 1 44 24" fill="none" stroke="#6366f1" stroke-width="4" stroke-linecap="round"/>'
    '</svg>'
    '<div style="margin-top:12px;color:#6366f1;font-size:14px;font-weight:500;">Vectorizing...</div>'
    '<div style="margin-top:4px;color:#aaa;font-size:12px;">This may take a moment</div>'
    '</div></div>'
    '<style>@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}</style>'
    '</div>'
)

PALETTE_PLACEHOLDER = (
    '<div style="color:#aaa;font-size:14px;">Color palette will appear here</div>'
)

CSS_PLACEHOLDER = (
    '<div style="color:#aaa;font-size:13px;font-family:monospace;">CSS variables will appear here</div>'
)

PRESETS = {
    "Photograph": (0, 0, 1.5, 80.0, 0.2, 8.0, 8.0, 0, 4000000, True),
    "Illustration": (16, 10, 1.0, 60.0, 0.5, 12.0, 12.0, 0, 6000000, True),
    "Line Art": (4, 5, 0.5, 120.0, 0.8, 5.0, 5.0, 0, 2000000, False),
}

# Toggle highlight JS (injected in head)
TOGGLE_HIGHLIGHT_JS = """
<script>
window._activePaletteColor = null;
window._togglePaletteHighlight = function(el, hexColor) {
    // If clicking same color, deactivate
    if (window._activePaletteColor === hexColor) {
        parent.highlightSvgColor(hexColor, false);
        el.querySelector('div').style.borderColor = 'transparent';
        window._activePaletteColor = null;
    } else {
        // Deactivate previous
        if (window._activePaletteColor) {
            parent.highlightSvgColor(window._activePaletteColor, false);
            var prev = document.querySelector('[data-color="' + window._activePaletteColor + '"]');
            if (prev) prev.querySelector('div').style.borderColor = 'transparent';
        }
        // Activate new
        parent.highlightSvgColor(hexColor, true);
        el.querySelector('div').style.borderColor = '#3b82f6';
        window._activePaletteColor = hexColor;
    }
};
</script>
"""

# ── Gradio app ────────────────────────────────────────────────────────────────

with gr.Blocks(title=BLOCK_TITLE) as demo:
    gr.Markdown(f"# {BLOCK_TITLE}")
    gr.Markdown(DESCRIPTION)

    # ── Presets row ──────────────────────────────────────────────────────────
    gr.Markdown("### Presets")
    with gr.Row():
        preset_photograph_btn = gr.Button("Photograph", size="sm")
        preset_illustration_btn = gr.Button("Illustration", size="sm")
        preset_lineart_btn = gr.Button("Line Art", size="sm")

    gr.Markdown("---")

    # ── Main input / preview row ──────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.Image(
                label="Input Image",
                type="filepath",
                height=400,
            )
            with gr.Row():
                vectorize_btn = gr.Button("Vectorize", variant="primary")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=2):
            svg_output = gr.HTML(value=SVG_PLACEHOLDER, label="SVG Preview")

            # SVG preview controls
            with gr.Row():
                zoom_in_btn = gr.Button("🔍+", size="sm")
                zoom_out_btn = gr.Button("🔍−", size="sm")
                fit_btn = gr.Button("⊡ Fit", size="sm")
                actual_btn = gr.Button("1:1", size="sm")
                fullscreen_btn = gr.Button("⛶", size="sm", elem_id="fullscreen-btn")
                bg_white_btn = gr.Button("⬜", size="sm")
                bg_black_btn = gr.Button("⬛", size="sm")
                bg_transparent_btn = gr.Button("▦", size="sm")

    # ── Comparison slider (Phase 3) ─────────────────────────────────────────
    gr.Markdown("### Before / After Comparison")
    with gr.Row():
        with gr.Column(scale=2):
            comparison_slider = gr.ImageSlider(
                label="Slide to compare — Left: Original  |  Right: Vectorized",
                slider_position=50,
                height=460,
            )
        with gr.Column(scale=1):
            # Compact metadata next to comparison
            metadata_output = gr.Textbox(
                label="Metadata",
                lines=6,
                interactive=False,
            )

    # ── Parameters ───────────────────────────────────────────────────────────
    gr.Markdown("### Configuration")

    with gr.Row():
        with gr.Column():
            num_colors = gr.Slider(
                label="Number of Colors (0 = auto)",
                minimum=0, maximum=256, value=0, step=1,
            )
            min_region_area = gr.Slider(
                label="Min Region Area (px²)",
                minimum=0, maximum=500, value=0, step=1,
            )
            curve_fit_error = gr.Slider(
                label="Curve Fit Error (px)",
                minimum=0.1, maximum=10.0, value=1.0, step=0.1,
            )
            corner_angle_threshold = gr.Slider(
                label="Corner Angle Threshold (°)",
                minimum=1.0, maximum=180.0, value=60.0, step=1.0,
            )
            smoothness = gr.Slider(
                label="Smoothness [0, 1]",
                minimum=0.0, maximum=1.0, value=0.3, step=0.01,
            )

        with gr.Column():
            smoothing_spatial = gr.Slider(
                label="Smoothing Spatial Radius",
                minimum=0.0, maximum=50.0, value=10.0, step=0.5,
            )
            smoothing_color = gr.Slider(
                label="Smoothing Color Radius",
                minimum=0.0, maximum=50.0, value=10.0, step=0.5,
            )
            upscale_short_edge = gr.Slider(
                label="Upscale Short Edge (0 = off)",
                minimum=0, maximum=2000, value=0, step=1,
            )
            max_working_pixels = gr.Slider(
                label="Max Working Pixels (0 = off)",
                minimum=0, maximum=50000000, value=4000000, step=100000,
            )
            enable_subpixel_refine = gr.Checkbox(
                label="Enable Sub-pixel Refinement",
                value=True,
            )

    # ── Results ─────────────────────────────────────────────────────────────
    gr.Markdown("### Result")

    # Hidden textbox that stores raw SVG — used by JS copy function
    svg_source_hidden = gr.Textbox(
        label="SVG Source",
        lines=6,
        visible=False,
        elem_id="svg-source-hidden",
    )

    with gr.Row():
        with gr.Column(scale=1):
            palette_output = gr.HTML(
                value=PALETTE_PLACEHOLDER,
                label="Color Palette",
            )
            palette_export = gr.Code(
                value="",
                language="css",
                label="Palette as CSS Variables",
                lines=6,
                visible=True,
            )
        with gr.Column(scale=1):
            with gr.Row():
                svg_download = gr.File(label="Download SVG", visible=True)
                png_download = gr.File(label="Download PNG (rendered)", visible=True)
            copy_svg_btn = gr.Button(
                "Copy SVG Source",
                size="sm",
                elem_id="copy-svg-btn",
            )

    # ── Event handlers ───────────────────────────────────────────────────────

    def _apply_preset(preset_name):
        vals = PRESETS[preset_name]
        return [gr.update(value=v) for v in vals]

    for preset_btn, preset_name in [
        (preset_photograph_btn, "Photograph"),
        (preset_illustration_btn, "Illustration"),
        (preset_lineart_btn, "Line Art"),
    ]:
        preset_btn.click(
            lambda n=preset_name: _apply_preset(n),
            outputs=[num_colors, min_region_area, curve_fit_error,
                     corner_angle_threshold, smoothness,
                     smoothing_spatial, smoothing_color,
                     upscale_short_edge, max_working_pixels, enable_subpixel_refine],
        )

    def on_vectorize(
        image_file, num_colors, min_region_area, curve_fit_error,
        corner_angle_threshold, smoothness, smoothing_spatial,
        smoothing_color, upscale_short_edge, max_working_pixels,
        enable_subpixel_refine,
    ):
        if image_file is None:
            gr.Info("Please upload an image first.")
            return (
                [SVG_PLACEHOLDER, "", PALETTE_PLACEHOLDER, "",
                 None, None, CSS_PLACEHOLDER, None]
            )

        file_size = Path(image_file).stat().st_size
        if file_size > 50 * 1024 * 1024:
            gr.Info(f"File too large ({file_size / 1024 / 1024:.1f} MB). Limit is 50 MB.")
            return (
                [SVG_PLACEHOLDER, "", PALETTE_PLACEHOLDER, "",
                 None, None, CSS_PLACEHOLDER, None]
            )

        try:
            (svg_content, metadata, palette_html,
             palette_css, png_bytes,
             original_image) = vectorize_image(
                image_file, num_colors, min_region_area, curve_fit_error,
                corner_angle_threshold, smoothness, smoothing_spatial,
                smoothing_color, upscale_short_edge, max_working_pixels,
                enable_subpixel_refine,
            )

            # Wrap SVG for display — scheduleSvgInit re-attaches wheel/panning
            wrapped = (
                '<div id="svg-preview-outer" style="width:100%;height:460px;display:flex;'
                'align-items:center;justify-content:center;'
                'background:#ffffff;box-sizing:border-box;overflow:auto;cursor:grab;" '
                'onload="if(window.scheduleSvgInit)scheduleSvgInit()">'
                '<div id="svg-preview-inner" style="max-width:100%;transform:translate(0px,0px) scale(1);transform-origin:center center;">'
                + svg_content +
                '</div></div>'
                '<script>scheduleSvgInit();</script>'
            )

            # Build image tuple for ImageSlider: (original_path, svg_png_path)
            if png_bytes is not None:
                # Save PNG to temp file for ImageSlider
                svg_png_tmpdir = Path(tempfile.gettempdir()) / "neroued-vectorizer"
                svg_png_tmpdir.mkdir(exist_ok=True)
                if not hasattr(on_vectorize, "_png_cleanup_done"):
                    for f in svg_png_tmpdir.glob("*.png"):
                        try:
                            f.unlink()
                        except OSError:
                            pass
                    on_vectorize._png_cleanup_done = True

                png_tmp = tempfile.NamedTemporaryFile(
                    mode='wb', suffix=".png", delete=False, dir=str(svg_png_tmpdir)
                )
                png_tmp.write(png_bytes)
                png_tmp.close()
                png_path = png_tmp.name
            else:
                png_path = None

            # Save SVG to temp file for download
            svg_tmpdir = Path(tempfile.gettempdir()) / "neroued-vectorizer"
            svg_tmpdir.mkdir(exist_ok=True)
            if not hasattr(on_vectorize, "_svg_cleanup_done"):
                for f in svg_tmpdir.glob("*.svg"):
                    try:
                        f.unlink()
                    except OSError:
                        pass
                on_vectorize._svg_cleanup_done = True

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".svg", delete=False, dir=str(svg_tmpdir)
            ) as f:
                f.write(svg_content)
                svg_path = f.name

            # Build ImageSlider value: tuple of (original, svg_png)
            if png_path and original_image:
                comparison_value = (original_image, png_path)
            else:
                comparison_value = None

            return (
                [wrapped, metadata, palette_html, palette_css, comparison_value,
                 svg_path, png_path, svg_content]
            )

        except Exception as exc:
            gr.Warning(f"Vectorization failed: {exc}")
            return (
                [SVG_PLACEHOLDER, f"Error: {exc}", PALETTE_PLACEHOLDER, "",
                 None, None, CSS_PLACEHOLDER, None]
            )

    def on_clear():
        return (
            [None, SVG_PLACEHOLDER, "", PALETTE_PLACEHOLDER, "",
             None, None, CSS_PLACEHOLDER, None]
        )

    # Main vectorize event with progress
    vectorize_btn.click(
        on_vectorize,
        inputs=[image_input, num_colors, min_region_area, curve_fit_error,
                corner_angle_threshold, smoothness, smoothing_spatial,
                smoothing_color, upscale_short_edge, max_working_pixels,
                enable_subpixel_refine],
        outputs=[svg_output, metadata_output, palette_output, palette_export,
                 comparison_slider, svg_download, png_download, svg_source_hidden],
        show_progress="full",
    )

    clear_btn.click(
        on_clear,
        outputs=[image_input, svg_output, metadata_output, palette_output,
                 palette_export, comparison_slider, svg_download, png_download, svg_source_hidden],
    )

    # Copy SVG button
    copy_svg_btn.click(
        None,
        js="() => { "
            "var src = document.getElementById('svg-source-hidden'); "
            "if (!src || !src.value) { return; } "
            "navigator.clipboard.writeText(src.value).then(function() { "
            "  var btn = document.getElementById('copy-svg-btn'); "
            "  if (btn) { btn.textContent = 'Copied!'; btn.style.color = '#22c55e'; "
            "    setTimeout(function(){ btn.textContent = 'Copy SVG Source'; btn.style.color = ''; }, 1500); } "
            "}).catch(function(e){ console.error(e); }); "
            "}",
    )

    # SVG preview controls
    zoom_in_btn.click(
        None,
        js="() => { "
            "var inner = document.getElementById('svg-preview-inner'); "
            "if (!inner) return; "
            "var cur = inner.style.transform.match(/scale\\(([\\d.]+)\\)/); "
            "var level = cur ? parseFloat(cur[1]) : 1; "
            "level = Math.min(20, level * 1.25); "
            "var m = inner.style.transform.match(/translate\\(([\\d.-]+)px, ([\\d.-]+)px\\)/); "
            "var tx = m ? m[1] : '0'; var ty = m ? m[2] : '0'; "
            "inner.style.transform = 'translate(' + tx + 'px, ' + ty + 'px) scale(' + level + ')'; "
            "inner.style.transformOrigin = 'center center'; "
            "}",
    )

    zoom_out_btn.click(
        None,
        js="() => { "
            "var inner = document.getElementById('svg-preview-inner'); "
            "if (!inner) return; "
            "var cur = inner.style.transform.match(/scale\\(([\\d.]+)\\)/); "
            "var level = cur ? parseFloat(cur[1]) : 1; "
            "level = Math.max(0.1, level / 1.25); "
            "var m = inner.style.transform.match(/translate\\(([\\d.-]+)px, ([\\d.-]+)px\\)/); "
            "var tx = m ? m[1] : '0'; var ty = m ? m[2] : '0'; "
            "inner.style.transform = 'translate(' + tx + 'px, ' + ty + 'px) scale(' + level + ')'; "
            "inner.style.transformOrigin = 'center center'; "
            "}",
    )

    fit_btn.click(
        None,
        js="() => { "
            "var outer = document.getElementById('svg-preview-outer'); "
            "var inner = document.getElementById('svg-preview-inner'); "
            "if (!outer || !inner) return; "
            "var svg = inner.querySelector('svg'); "
            "if (!svg) return; "
            "var vb = svg.getAttribute('viewBox'); "
            "var w, h; "
            "if (vb) { var p = vb.split(' ').map(Number); w = p[2]; h = p[3]; } "
            "else { w = parseFloat(svg.getAttribute('width')) || inner.offsetWidth; "
            "       h = parseFloat(svg.getAttribute('height')) || inner.offsetHeight; } "
            "var scaleX = (outer.offsetWidth - 16) / w; "
            "var scaleY = (outer.offsetHeight - 16) / h; "
            "var scale = Math.min(scaleX, scaleY, 2); "
            "inner.style.transform = 'translate(0px,0px) scale(' + scale + ')'; "
            "inner.style.transformOrigin = 'center center'; "
            "}",
    )

    actual_btn.click(
        None,
        js="() => { "
            "var inner = document.getElementById('svg-preview-inner'); "
            "if (!inner) return; "
            "inner.style.transform = 'translate(0px,0px) scale(1)'; "
            "inner.style.transformOrigin = 'center center'; "
            "}",
    )

    fullscreen_btn.click(
        None,
        js="() => { "
            "var outer = document.getElementById('svg-preview-outer'); "
            "if (!outer) return; "
            "if (!document.fullscreenElement) { "
            "  outer.requestFullscreen().catch(function(e){ console.error(e); }); "
            "} else { "
            "  document.exitFullscreen(); "
            "} "
            "}",
    )

    # Background buttons
    for bg_name, bg_val in [("white", "#ffffff"), ("black", "#111111")]:
        bg_btn = bg_white_btn if bg_name == "white" else bg_black_btn
        bg_btn.click(
            None,
            js=f"() => {{ "
                f"var outer = document.getElementById('svg-preview-outer'); "
                f"if (outer) outer.style.background = '{bg_val}'; "
                f"}}",
        )

    bg_transparent_btn.click(
        None,
        js="() => { "
            "var outer = document.getElementById('svg-preview-outer'); "
            "if (outer) outer.style.background = "
            "'repeating-conic-gradient(#e0e0e0 0% 25%, #fff 0% 50%) 50%/16px 16px'; "
            "}",
    )


if __name__ == "__main__":
    demo.launch(
        server_port=7861,
        server_name="0.0.0.0",
        head=SVG_PREVIEW_JS + COPY_FEEDBACK_JS + TOGGLE_HIGHLIGHT_JS,
    )
