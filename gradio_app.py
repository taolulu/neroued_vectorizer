"""Gradio Web UI for neroued-vectorizer service — visual & responsive overhaul."""

from __future__ import annotations
import io
import re
import tempfile
from pathlib import Path

from PIL import Image
import cairosvg
import gradio as gr
from gradio.themes import Default as DefaultTheme

import neroued_vectorizer as nv

# ── Module-level session history (survives across event calls) ────────────────
_session_history: list[dict] = []

# ── SVG Preview JavaScript ───────────────────────────────────────────────────

SVG_PREVIEW_JS = """
<script>
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
                p.style.filter = 'drop-shadow(0 0 6px ' + hexColor + ')';
                p.style.strokeWidth = '2';
            } else {
                p.style.filter = '';
                p.style.strokeWidth = '';
            }
        }
    });
}

function resetSvgHighlight() {
    var inner = document.getElementById('svg-preview-inner');
    if (!inner) return;
    var paths = inner.querySelectorAll('path, rect, circle, ellipse, polygon, polyline');
    paths.forEach(function(p) {
        p.style.filter = '';
        p.style.strokeWidth = '';
    });
}

// ── Save / load custom presets via localStorage ────────────────────────────
function saveCustomPreset(name, vals) {
    try {
        var raw = localStorage.getItem('neroued-presets') || '{}';
        var presets = JSON.parse(raw);
        presets[name] = vals;
        localStorage.setItem('neroued-presets', JSON.stringify(presets));
        return true;
    } catch(e) { return false; }
}

function loadCustomPresets() {
    try {
        var raw = localStorage.getItem('neroued-presets') || '{}';
        return JSON.parse(raw);
    } catch(e) { return {}; }
}

function deleteCustomPreset(name) {
    try {
        var raw = localStorage.getItem('neroued-presets') || '{}';
        var presets = JSON.parse(raw);
        delete presets[name];
        localStorage.setItem('neroued-presets', JSON.stringify(presets));
        return true;
    } catch(e) { return false; }
}

function getPresetVals() {
    var numColors = document.getElementById('num-colors-slider')?.value || 0;
    var minRegion = document.getElementById('min-region-slider')?.value || 0;
    var curveError = document.getElementById('curve-error-slider')?.value || 1.0;
    var cornerAngle = document.getElementById('corner-angle-slider')?.value || 60;
    var smoothness = document.getElementById('smoothness-slider')?.value || 0.3;
    var smoothSpatial = document.getElementById('smooth-spatial-slider')?.value || 10;
    var smoothColor = document.getElementById('smooth-color-slider')?.value || 10;
    var upscale = document.getElementById('upscale-slider')?.value || 0;
    var maxPixels = document.getElementById('max-pixels-slider')?.value || 4000000;
    var subpixel = document.getElementById('subpixel-checkbox')?.checked || false;
    return [parseInt(numColors), parseInt(minRegion), parseFloat(curveError),
            parseFloat(cornerAngle), parseFloat(smoothness), parseFloat(smoothSpatial),
            parseFloat(smoothColor), parseInt(upscale), parseInt(maxPixels), subpixel];
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
            btn.textContent = '已复制！';
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

def render_svg_to_png_bytes(svg_content: str, width: int, height: int) -> bytes:
    """Render SVG content to PNG bytes using cairosvg."""
    return cairosvg.svg2png(
        bytestring=svg_content.encode('utf-8'),
        output_width=width,
        output_height=height,
    )


# ── File info on upload ───────────────────────────────────────────────────────

def build_file_info_html(image_file: str | None) -> str:
    """Build HTML showing image dimensions, file size, and mode."""
    if image_file is None:
        return (
            '<div style="color:#9090b0;font-size:12px;font-family:monospace;padding:4px 0;">'
            '未上传图片</div>'
        )
    try:
        stat = Path(image_file).stat()
        file_size_kb = stat.st_size / 1024
        if file_size_kb >= 1024:
            size_str = f"{file_size_kb/1024:.1f} MB"
        else:
            size_str = f"{file_size_kb:.0f} KB"

        with Image.open(image_file) as img:
            width, height = img.size
            mode = img.mode

        mode_map = {"RGB": "彩色 (RGB)", "RGBA": "彩色+透明 (RGBA)", "L": "灰度 (L)", "1": "黑白 (1-bit)", "P": "调色板 (P)"}
        mode_label = mode_map.get(mode, f"{mode}")

        return (
            f'<div style="display:flex;gap:16px;flex-wrap:wrap;padding:6px 0;">'
            f'<span style="color:#9090b0;font-size:12px;font-family:monospace;">'
            f'📐 {width} × {height} px</span>'
            f'<span style="color:#9090b0;font-size:12px;font-family:monospace;">'
            f'💾 {size_str}</span>'
            f'<span style="color:#9090b0;font-size:12px;font-family:monospace;">'
            f'🎨 {mode_label}</span>'
            f'</div>'
        )
    except Exception as e:
        return (
            '<div style="color:#9090b0;font-size:12px;font-family:monospace;padding:4px 0;">'
            f'无法读取图片信息: {e}</div>'
        )


# ── Vectorization (with progress) ───────────────────────────────────────────

def vectorize_image_gen(
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
):
    """Generator: vectorize with per-stage progress yields.

    Yields None placeholders for each stage; the caller consumes them
    to drive Gradio's progress bar via show_progress="full".
    """

    # Stage 1: configure
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
    yield None

    # Stage 2: core vectorization
    result = nv.vectorize(image_file, config)
    yield None

    # Stage 3: render PNG preview
    png_bytes = None
    try:
        png_bytes = render_svg_to_png_bytes(result.svg_content, result.width, result.height)
    except Exception as e:
        print(f"[Warning] Could not render SVG to PNG: {e}")
    yield None

    # Done
    yield (
        result.svg_content,
        png_bytes,
        image_file,
    )


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
    """Vectorize an image (sync version, no progress)."""
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

    png_bytes = None
    try:
        png_bytes = render_svg_to_png_bytes(result.svg_content, result.width, result.height)
    except Exception as e:
        print(f"[Warning] Could not render SVG to PNG: {e}")

    return (
        result.svg_content,
        png_bytes,
        image_file,
    )


# ── App constants ─────────────────────────────────────────────────────────────

BLOCK_TITLE = "neroued 矢量化器"
DESCRIPTION = (
    "高质量栅格转 SVG 矢量化工具。上传图片、调整参数，点击矢量化即可生成 SVG。"
)

PRESETS = {
    "照片": {
        "label": "照片",
        "description": "适合人像、风景等照片级图片，自动保留丰富色彩",
        "icon": "📷",
        "params": (0, 0, 1.5, 80.0, 0.2, 8.0, 8.0, 0, 4000000, True),
    },
    "插画": {
        "label": "插画",
        "description": "适合动漫、游戏原画等插画类图片，平衡细节与文件大小",
        "icon": "🎨",
        "params": (16, 10, 1.0, 60.0, 0.5, 12.0, 12.0, 0, 6000000, True),
    },
    "线稿": {
        "label": "线稿",
        "description": "适合黑白线条画、图标、图纸，无色彩保真",
        "icon": "✏️",
        "params": (4, 5, 0.5, 120.0, 0.8, 5.0, 5.0, 0, 2000000, False),
    },
}

# ── Custom CSS ─────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
:root {
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --surface: #1e1e2e;
    --surface-2: #28283a;
    --surface-3: #313147;
    --border: #3a3a5c;
    --text: #e2e2f0;
    --text-muted: #9090b0;
    --accent: #22c55e;
    --danger: #ef4444;
}

body, body.gradio-mode {
    background: var(--surface) !important;
}

/* Section dividers */
.section-header {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 0 0 8px 0;
}

/* Parameter accordion */
.param-section .gr-accordion-header {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.param-section .gr-accordion-header:hover {
    background: var(--surface-3) !important;
}

/* Slider track styling */
input[type="range"] {
    accent-color: var(--primary) !important;
}

/* Download buttons */
.download-btn {
    background: var(--surface-3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    transition: background 0.2s, border-color 0.2s !important;
}
.download-btn:hover {
    background: var(--primary) !important;
    border-color: var(--primary-hover) !important;
}

/* Comparison slider container */
comparison-container {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
}

/* Hide default gradio section headers */
.gr-form {
    background: transparent !important;
}

/* History items */
.history-item {
    cursor: pointer;
    padding: 6px 8px;
    margin-bottom: 4px;
    background: #313147;
    border-radius: 6px;
    font-size: 12px;
    font-family: monospace;
    color: #e2e2f0;
    transition: background 0.15s;
}
.history-item:hover {
    background: #3a3a5c;
}

/* Radio preset styling */
.preset-radio-label {
    font-weight: 600;
    color: var(--text);
}
.preset-radio-description {
    font-size: 11px;
    color: var(--text-muted);
}

/* Responsive: tablet and below */
@media (max-width: 900px) {
    /* Stack layout is natural in single column */
}
"""

# ── Gradio app ────────────────────────────────────────────────────────────────

with gr.Blocks(title=BLOCK_TITLE) as demo:

    gr.Markdown(f"# {BLOCK_TITLE}")
    gr.Markdown(DESCRIPTION)

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 1 — Always expanded: upload, vectorize, preview, download
    # ═══════════════════════════════════════════════════════════════════════

    with gr.Group():
        gr.Markdown('<p class="section-header">📤 上传图片</p>')
        image_input = gr.Image(
            label="",
            type="filepath",
            height=320,
        )

        # File info display (updated on upload)
        file_info_output = gr.HTML(
            value='<div style="color:#9090b0;font-size:12px;font-family:monospace;padding:4px 0;">'
                  '未上传图片</div>',
            elem_id="file-info-display",
        )

        with gr.Row():
            vectorize_btn = gr.Button("✨ 矢量化", variant="primary", size="lg")
            clear_btn = gr.Button("🗑 清空", size="lg")

    gr.Markdown('<p class="section-header">🔍 效果对比</p>')
    comparison_slider = gr.ImageSlider(
        label="拖动滑块 — 左侧：原图  |  右侧：矢量结果",
        slider_position=50,
        height=400,
    )

    # Hidden SVG source for JS copy
    svg_source_hidden = gr.Textbox(
        visible=False,
        elem_id="svg-source-hidden",
    )

    gr.Markdown('<p class="section-header">📊 结果</p>')

    gr.Markdown("#### 💾 导出")
    with gr.Row():
        svg_download = gr.File(label="下载 SVG")
        png_download = gr.File(label="下载 PNG")

    with gr.Row():
        copy_svg_btn = gr.Button(
            "📋 复制 SVG 源码",
            size="sm",
            elem_id="copy-svg-btn",
        )

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 2 — Collapsed: preset radio + all parameter sliders in one accordion
    # ═══════════════════════════════════════════════════════════════════════

    with gr.Group(elem_classes=["param-section"]):
        with gr.Accordion("⚙️ 参数配置", open=False, elem_classes=["param-accordion"]):
            # ── Preset selection (native Gradio Radio) ──────────────────────
            gr.Markdown('<p class="section-header">⚡ 预设</p>')
            preset_choices = {
                "照片 (📷 适合人像、风景)": "照片",
                "插画 (🎨 适合动漫、游戏原画)": "插画",
                "线稿 (✏️ 适合黑白线条、图标)": "线稿",
            }
            preset_radio = gr.Radio(
                choices=list(preset_choices.keys()),
                value=None,
                label="",
                info="选择一个预设以快速填充参数，或手动调整下方滑块",
                elem_id="preset-radio",
            )

            gr.Markdown("---")

            # ── All parameter sliders in one accordion ─────────────────────
            gr.Markdown('<p class="section-header">🎨 色彩与区域</p>')
            num_colors = gr.Slider(
                label="颜色数量（0 = 自动）",
                info="输出 SVG 中的最大颜色数。设为 0 则由算法自动判断适合的数量。",
                minimum=0, maximum=256, value=0, step=1,
                elem_id="num-colors-slider",
            )
            min_region_area = gr.Slider(
                label="最小区域面积（px²）",
                info="小于此面积的色块会被忽略并合并到周围区域。可去除噪点。",
                minimum=0, maximum=500, value=0, step=1,
                elem_id="min-region-slider",
            )
            enable_subpixel_refine = gr.Checkbox(
                label="启用亚像素细化",
                info="在像素级别优化边缘，减少锯齿和色差。略微增加处理时间。",
                value=True,
                elem_id="subpixel-checkbox",
            )

            gr.Markdown('<p class="section-header">📐 曲线与形状</p>')
            curve_fit_error = gr.Slider(
                label="曲线拟合误差（px）",
                info="允许的曲线与原始边缘最大偏差（像素）。越大曲线越平滑简洁，越小越接近原始形状。",
                minimum=0.1, maximum=10.0, value=1.0, step=0.1,
                elem_id="curve-error-slider",
            )
            corner_angle_threshold = gr.Slider(
                label="拐角角度阈值（°）",
                info="小于此角度的角点会被当作尖角保留；越大越容易将尖锐转角变成圆弧。",
                minimum=1.0, maximum=180.0, value=60.0, step=1.0,
                elem_id="corner-angle-slider",
            )
            smoothness = gr.Slider(
                label="平滑度 [0, 1]",
                info="贝塞尔曲线平滑系数。0 保持原始锯齿，1 最平滑但可能失真。",
                minimum=0.0, maximum=1.0, value=0.3, step=0.01,
                elem_id="smoothness-slider",
            )

            gr.Markdown('<p class="section-header">🔧 高级选项</p>')
            smoothing_spatial = gr.Slider(
                label="平滑空间半径",
                info="空间高斯模糊的半径（像素）。值越大颜色过渡越柔和。",
                minimum=0.0, maximum=50.0, value=10.0, step=0.5,
                elem_id="smooth-spatial-slider",
            )
            smoothing_color = gr.Slider(
                label="平滑色彩半径",
                info="色彩空间高斯模糊半径。值越大相近颜色越容易融合。",
                minimum=0.0, maximum=50.0, value=10.0, step=0.5,
                elem_id="smooth-color-slider",
            )
            upscale_short_edge = gr.Slider(
                label="放大短边（0 = 关闭）",
                info="在处理前先将图片短边放大到此像素值再矢量化。可提升小图的细节保留。",
                minimum=0, maximum=2000, value=0, step=1,
                elem_id="upscale-slider",
            )
            max_working_pixels = gr.Slider(
                label="最大处理像素（0 = 关闭）",
                info="处理过程中工作区的最大像素数。超过则等比缩小。可防止内存溢出。",
                minimum=0, maximum=50000000, value=4000000, step=100000,
                elem_id="max-pixels-slider",
            )

    # ── Session history ───────────────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown('<p class="section-header">📜 本次会话历史</p>')
    history_output = gr.HTML(
        value='<div style="color:#9090b0;font-size:12px;padding:4px 0;">尚无历史记录</div>',
        elem_id="history-list",
    )

    # ── Event handlers ───────────────────────────────────────────────────────

    def _apply_preset(preset_name: str):
        """Return updated values for all parameter components from preset."""
        preset_map = {
            "照片": PRESETS["照片"]["params"],
            "插画": PRESETS["插画"]["params"],
            "线稿": PRESETS["线稿"]["params"],
        }
        if preset_name in preset_map:
            vals = preset_map[preset_name]
            return [gr.update(value=v) for v in vals]
        return [gr.no_update()] * 10

    # Preset JS — simplified (preset card functions removed, now using native Gradio Radio)
    preset_js = ""


    def on_upload(image_file):
        """Update file info display when an image is uploaded."""
        return build_file_info_html(image_file)

    def on_vectorize(
        image_file, num_colors, min_region_area, curve_fit_error,
        corner_angle_threshold, smoothness, smoothing_spatial,
        smoothing_color, upscale_short_edge, max_working_pixels,
        enable_subpixel_refine,
    ):
        global _session_history

        if image_file is None:
            gr.Info("请先上传一张图片。")
            return (
                [gr.no_update(), None, None, None]
            )

        file_size = Path(image_file).stat().st_size
        if file_size > 50 * 1024 * 1024:
            gr.Info(f"文件过大（{file_size / 1024 / 1024:.1f} MB）。限制为 50 MB。")
            return (
                [gr.no_update(), None, None, None]
            )

        params = {
            "num_colors": num_colors,
            "min_region_area": min_region_area,
            "curve_fit_error": curve_fit_error,
            "corner_angle_threshold": corner_angle_threshold,
            "smoothness": smoothness,
            "smoothing_spatial": smoothing_spatial,
            "smoothing_color": smoothing_color,
            "upscale_short_edge": upscale_short_edge,
            "max_working_pixels": max_working_pixels,
            "enable_subpixel_refine": enable_subpixel_refine,
        }

        try:
            # Iterate through generator (each yield updates the Gradio progress bar)
            gen = vectorize_image_gen(
                image_file, num_colors, min_region_area, curve_fit_error,
                corner_angle_threshold, smoothness, smoothing_spatial,
                smoothing_color, upscale_short_edge, max_working_pixels,
                enable_subpixel_refine,
            )

            # Consume all intermediate None yields (progress updates)
            result = None
            for step in gen:
                if step is not None:
                    result = step

            if result is None:
                raise RuntimeError("矢量化未返回结果")

            (svg_content, png_bytes,
             original_image) = result

            if png_bytes is not None:
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

            if png_path and original_image:
                comparison_value = (original_image, png_path)
            else:
                comparison_value = None

            # Build history entry
            label = f"#{len(_session_history)+1} {num_colors}色"
            _session_history.append({"label": label, "params": params})

            return (
                [comparison_value, svg_path, png_path, svg_content]
            )

        except Exception as exc:
            gr.Warning(f"矢量化失败：{exc}")
            return (
                [gr.no_update(), None, None, None]
            )

    def build_history_html() -> str:
        """Build clickable HTML list of history entries (reads module-level _session_history)."""
        global _session_history
        if not _session_history:
            return '<div style="color:#9090b0;font-size:12px;padding:4px 0;">尚无历史记录</div>'
        items = []
        for i, entry in enumerate(reversed(_session_history)):
            items.append(
                f'<div class="history-item" onclick="loadHistory({i})" '
                f'style="cursor:pointer;padding:6px 8px;margin-bottom:4px;'
                f'background:#313147;border-radius:6px;font-size:12px;font-family:monospace;'
                f'color:#e2e2f0;transition:background 0.15s;" '
                f'onmouseover="this.style.background=\'#3a3a5c\'" '
                f'onmouseout="this.style.background=\'#313147\'">'
                f'↩ {entry["label"]}</div>'
            )
        return "\n".join(items)

    def on_clear():
        global _session_history
        _session_history = []
        return (
            [None, None, None, None]
        )

    def on_history_select(history_index: int):
        """Load parameters from a history entry back into the sliders."""
        global _session_history
        if history_index is None or history_index >= len(_session_history):
            return [gr.no_update()] * 10
        entry = _session_history[history_index]
        p = entry["params"]
        return [
            gr.update(value=p["num_colors"]),
            gr.update(value=p["min_region_area"]),
            gr.update(value=p["curve_fit_error"]),
            gr.update(value=p["corner_angle_threshold"]),
            gr.update(value=p["smoothness"]),
            gr.update(value=p["smoothing_spatial"]),
            gr.update(value=p["smoothing_color"]),
            gr.update(value=p["upscale_short_edge"]),
            gr.update(value=p["max_working_pixels"]),
            gr.update(value=p["enable_subpixel_refine"]),
        ]

    def on_preset_radio_change(preset_key: str | None):
        """When user selects a preset from the Radio, update all parameter sliders."""
        if preset_key is None:
            return [gr.no_update()] * 10
        if preset_key not in PRESETS:
            return [gr.no_update()] * 10
        vals = PRESETS[preset_key]["params"]
        return [gr.update(value=v) for v in vals]

    # Main vectorize event
    vectorize_btn.click(
        on_vectorize,
        inputs=[image_input, num_colors, min_region_area, curve_fit_error,
                corner_angle_threshold, smoothness, smoothing_spatial,
                smoothing_color, upscale_short_edge, max_working_pixels,
                enable_subpixel_refine],
        outputs=[comparison_slider, svg_download, png_download, svg_source_hidden],
    ).then(
        fn=build_history_html,
        inputs=None,
        outputs=[history_output],
    )

    clear_btn.click(
        on_clear,
        outputs=[image_input, comparison_slider, svg_download, png_download, svg_source_hidden],
    ).then(
        fn=build_history_html,
        inputs=None,
        outputs=[history_output],
    )

    # Upload → file info
    image_input.upload(
        on_upload,
        inputs=[image_input],
        outputs=[file_info_output],
    )

    # Preset radio → update all sliders
    preset_radio.change(
        fn=on_preset_radio_change,
        inputs=[preset_radio],
        outputs=[num_colors, min_region_area, curve_fit_error,
                  corner_angle_threshold, smoothness, smoothing_spatial,
                  smoothing_color, upscale_short_edge, max_working_pixels,
                  enable_subpixel_refine],
    )

    # Hidden index carrier for history selection
    history_index_input = gr.Number(visible=False, value=None, container=False)

    def on_history_index_change(index):
        """Triggered when JS writes the selected history index to the hidden field."""
        if index is None:
            return [gr.no_update()] * 10
        return on_history_select(int(index))

    history_index_input.change(
        fn=on_history_index_change,
        inputs=[history_index_input],
        outputs=[num_colors, min_region_area, curve_fit_error,
                 corner_angle_threshold, smoothness, smoothing_spatial,
                 smoothing_color, upscale_short_edge, max_working_pixels,
                 enable_subpixel_refine],
        queue=False,
    )

    # History JS function
    history_js = """
    <script>
    function loadHistory(idx) {
        // Write the clicked history index to the hidden Number field
        var inp = document.querySelector('input[type="number"][aria-label=""]');
        if (!inp) return;
        // Gradio stores the hidden number in a zero-sized input
        var allInputs = document.querySelectorAll('input[type="number"]');
        for (var i = 0; i < allInputs.length; i++) {
            if (allInputs[i].offsetParent === null || allInputs[i].style.display === 'none') {
                allInputs[i].value = idx;
                allInputs[i].dispatchEvent(new Event('input', {bubbles: true}));
                allInputs[i].dispatchEvent(new Event('change', {bubbles: true}));
                break;
            }
        }
    }
    </script>
    """

    # Copy SVG button
    copy_svg_btn.click(
        None,
        js="() => { "
            "var src = document.getElementById('svg-source-hidden'); "
            "if (!src || !src.value) { return; } "
            "navigator.clipboard.writeText(src.value).then(function() { "
            "  var btn = document.getElementById('copy-svg-btn'); "
            "  if (btn) { btn.textContent = '✅ 已复制！'; btn.style.color = '#22c55e'; "
            "    setTimeout(function(){ btn.textContent = '📋 复制 SVG 源码'; btn.style.color = ''; }, 1500); } "
            "}).catch(function(e){ console.error(e); }); "
            "}",
    )


if __name__ == "__main__":
    demo.launch(
        server_port=7861,
        server_name="0.0.0.0",
        theme=gr.themes.Default(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
            font=("Inter", "system-ui", "sans-serif"),
        ),
        css=CUSTOM_CSS,
        head=SVG_PREVIEW_JS + COPY_FEEDBACK_JS + preset_js + history_js,
    )
