"""Gradio Web UI for neroued-vectorizer service — visual & responsive overhaul."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import cairosvg
import gradio as gr
from gradio.themes import Default as DefaultTheme

import neroued_vectorizer as nv

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

def parse_svg_color_stats(svg_content: str) -> dict[str, dict]:
    """Parse SVG and return per-color path count and percentage."""
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
        color: {'count': count, 'pct': round(count / total * 100, 1)}
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
    """Build interactive palette HTML with usage % and hover highlight."""
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
            f'onmouseover="highlightSvgColor(\'{hex_color}\', true)" '
            f'onmouseout="highlightSvgColor(\'{hex_color}\', false)" '
            f'style="display:inline-flex;flex-direction:column;align-items:center;'
            f'margin:4px;cursor:pointer;" '
            f'title="{hex_color} — {stat["count"]} shapes ({stat["pct"]}%)">'
            f'<div style="width:52px;height:52px;background:{hex_color};'
            f'border:2px solid transparent;border-radius:8px;'
            f'box-shadow:0 2px 8px rgba(0,0,0,0.35);'
            f'transition:border-color 0.15s,transform 0.15s;" '
            f'onmouseover="this.style.borderColor=\'#3b82f6\';this.style.transform=\'scale(1.12)\'" '
            f'onmouseout="this.style.borderColor=\'transparent\';this.style.transform=\'scale(1)\'">'
            f'<span style="display:block;padding-top:16px;font-size:12px;'
            f'font-weight:bold;color:{text_color};font-family:monospace;">{i+1}</span></div>'
            f'<span style="font-size:10px;color:#666;margin-top:3px;font-family:monospace;">'
            f'{stat["pct"]}%</span></div>'
        )
    return (
        f'<div id="palette-container" style="display:flex;flex-wrap:wrap;gap:2px;padding:4px 0;">'
        + ''.join(swatches)
        + '</div>'
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
        f"宽度：{result.width} px\n"
        f"高度：{result.height} px\n"
        f"形状数：{result.num_shapes}\n"
        f"颜色数：{result.resolved_num_colors}\n"
        + "\n".join(
            f"  {i+1}. #{int(c.r*255):02x}{int(c.g*255):02x}{int(c.b*255):02x}"
            for i, c in enumerate(result.palette)
        )
    )

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

BLOCK_TITLE = "neroued 矢量化器"
DESCRIPTION = (
    "高质量栅格转 SVG 矢量化工具。上传图片、调整参数，点击矢量化即可生成 SVG。"
)

PALETTE_PLACEHOLDER = (
    '<div style="color:#aaa;font-size:14px;padding:8px;">调色板将在矢量化后显示</div>'
)

CSS_PLACEHOLDER = (
    '<div style="color:#aaa;font-size:13px;font-family:monospace;padding:8px;">CSS 变量将在此处显示</div>'
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

/* Card-style preset buttons */
.preset-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 12px 16px;
    border: 1.5px solid var(--border);
    border-radius: 12px;
    background: var(--surface-2);
    cursor: pointer;
    transition: border-color 0.2s, transform 0.15s, box-shadow 0.2s;
    min-width: 100px;
    text-align: center;
    user-select: none;
}
.preset-card:hover {
    border-color: var(--primary);
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.25);
}
.preset-card .preset-icon { font-size: 24px; margin-bottom: 4px; }
.preset-card .preset-label { font-weight: 600; color: var(--text); font-size: 14px; }
.preset-card .preset-desc { font-size: 11px; color: var(--text-muted); margin-top: 2px; }

/* Save preset button */
.save-preset-btn {
    border: 1.5px dashed var(--border);
    background: transparent;
    border-radius: 12px;
    color: var(--text-muted);
    transition: border-color 0.2s, color 0.2s;
}
.save-preset-btn:hover {
    border-color: var(--accent);
    color: var(--accent);
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
. param-section .gr-accordion-header {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
. param-section .gr-accordion-header:hover {
    background: var(--surface-3) !important;
}

/* Slider track styling */
input[type="range"] {
    accent-color: var(--primary) !important;
}

/* Metadata box */
.metadata-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
    color: var(--text-muted);
    line-height: 1.6;
}

/* Palette swatches */
.palette-swatch {
    transition: transform 0.15s;
}
.palette-swatch:hover {
    transform: translateY(-2px);
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

/* Responsive: tablet and below */
@media (max-width: 900px) {
    .main-layout {
        flex-direction: column !important;
    }
    .left-panel, .right-panel {
        width: 100% !important;
        max-width: 100% !important;
    }
    /* Stack parameter accordions */
    .param-accordion {
        margin-bottom: 8px;
    }
}

/* Mobile tweaks */
@media (max-width: 600px) {
    .preset-cards-row {
        gap: 8px !important;
    }
    .preset-card {
        min-width: 80px;
        padding: 8px 10px;
    }
    .preset-card .preset-desc {
        display: none;
    }
    .preset-card .preset-icon { font-size: 20px; }
}
"""

# ── Gradio app ────────────────────────────────────────────────────────────────

with gr.Blocks(title=BLOCK_TITLE) as demo:

    gr.Markdown(f"# {BLOCK_TITLE}")
    gr.Markdown(DESCRIPTION)

    # ── Presets row (card-style) ─────────────────────────────────────────────
    gr.Markdown('<p class="section-header">⚡ 预设</p>')
    with gr.Row(elem_classes=["preset-cards-row"]):
        for preset_key, preset_val in PRESETS.items():
            with gr.Column():
                gr.HTML(
                    f'<div class="preset-card" id="preset-{preset_key}" '
                    f'onclick="applyPreset(\'{preset_key}\')">'
                    f'<span class="preset-icon">{preset_val["icon"]}</span>'
                    f'<span class="preset-label">{preset_val["label"]}</span>'
                    f'<span class="preset-desc">{preset_val["description"]}</span>'
                    f'</div>'
                )

        # Save custom preset button
        with gr.Column():
            gr.HTML(
                '<div class="preset-card save-preset-btn" id="save-preset-btn" '
                'onclick="openSavePresetDialog()">'
                '<span class="preset-icon">💾</span>'
                '<span class="preset-label">保存当前</span>'
                '<span class="preset-desc">保存为自定义预设</span>'
                '</div>'
            )

    gr.Markdown("---")

    # ── Main two-column layout ──────────────────────────────────────────────
    with gr.Row(elem_classes=["main-layout"]):

        # ── LEFT: Upload + Preview ──────────────────────────────────────────
        with gr.Column(scale=1, elem_classes=["left-panel"]):
            gr.Markdown('<p class="section-header">📤 上传图片</p>')
            image_input = gr.Image(
                label="",
                type="filepath",
                height=320,
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

        # ── RIGHT: Parameters + Results ───────────────────────────────────
        with gr.Column(scale=1, elem_classes=["right-panel"]):

            # ── Parameters accordion ─────────────────────────────────────────
            gr.Markdown('<p class="section-header">⚙️ 参数配置</p>')

            with gr.Accordion("🎨 色彩与区域", open=True, elem_classes=["param-accordion"]):
                num_colors = gr.Slider(
                    label="颜色数量（0 = 自动）",
                    minimum=0, maximum=256, value=0, step=1,
                    elem_id="num-colors-slider",
                )
                min_region_area = gr.Slider(
                    label="最小区域面积（px²）",
                    minimum=0, maximum=500, value=0, step=1,
                    elem_id="min-region-slider",
                )
                enable_subpixel_refine = gr.Checkbox(
                    label="启用亚像素细化",
                    value=True,
                    elem_id="subpixel-checkbox",
                )

            with gr.Accordion("📐 曲线与形状", open=True, elem_classes=["param-accordion"]):
                curve_fit_error = gr.Slider(
                    label="曲线拟合误差（px）",
                    minimum=0.1, maximum=10.0, value=1.0, step=0.1,
                    elem_id="curve-error-slider",
                )
                corner_angle_threshold = gr.Slider(
                    label="拐角角度阈值（°）",
                    minimum=1.0, maximum=180.0, value=60.0, step=1.0,
                    elem_id="corner-angle-slider",
                )
                smoothness = gr.Slider(
                    label="平滑度 [0, 1]",
                    minimum=0.0, maximum=1.0, value=0.3, step=0.01,
                    elem_id="smoothness-slider",
                )

            with gr.Accordion("🔧 高级选项", open=False, elem_classes=["param-accordion"]):
                smoothing_spatial = gr.Slider(
                    label="平滑空间半径",
                    minimum=0.0, maximum=50.0, value=10.0, step=0.5,
                    elem_id="smooth-spatial-slider",
                )
                smoothing_color = gr.Slider(
                    label="平滑色彩半径",
                    minimum=0.0, maximum=50.0, value=10.0, step=0.5,
                    elem_id="smooth-color-slider",
                )
                upscale_short_edge = gr.Slider(
                    label="放大短边（0 = 关闭）",
                    minimum=0, maximum=2000, value=0, step=1,
                    elem_id="upscale-slider",
                )
                max_working_pixels = gr.Slider(
                    label="最大处理像素（0 = 关闭）",
                    minimum=0, maximum=50000000, value=4000000, step=100000,
                    elem_id="max-pixels-slider",
                )

            gr.Markdown("---")

            # ── Results ──────────────────────────────────────────────────────
            gr.Markdown('<p class="section-header">📊 结果</p>')

            metadata_output = gr.Textbox(
                label="元数据",
                lines=5,
                interactive=False,
                elem_classes=["metadata-box"],
            )

            gr.Markdown("#### 🎨 调色板")
            palette_output = gr.HTML(value=PALETTE_PLACEHOLDER)
            palette_export = gr.Code(
                value="",
                language="css",
                label="CSS 变量",
                lines=5,
            )

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

    # Preset card JS injection — apply preset values via JS → then sync to Gradio
    preset_js = """
    <script>
    function applyPreset(name) {
        var presets = {
            '照片': [0, 0, 1.5, 80.0, 0.2, 8.0, 8.0, 0, 4000000, true],
            '插画': [16, 10, 1.0, 60.0, 0.5, 12.0, 12.0, 0, 6000000, true],
            '线稿': [4, 5, 0.5, 120.0, 0.8, 5.0, 5.0, 0, 2000000, false],
        };
        var vals = presets[name];
        if (!vals) return;

        // Update native range/checkbox inputs directly for instant UI feedback
        var ids = ['num-colors-slider','min-region-slider','curve-error-slider',
                   'corner-angle-slider','smoothness-slider','smooth-spatial-slider',
                   'smooth-color-slider','upscale-slider','max-pixels-slider','subpixel-checkbox'];
        ids.forEach(function(id, i) {
            var el = document.getElementById(id) || document.querySelector('[id="'+id+'"] input');
            if (el) {
                if (el.type === 'range') el.value = vals[i];
                else if (el.type === 'checkbox') el.checked = vals[i];
            }
        });

        // Dispatch input events so Gradio state updates
        ids.forEach(function(id) {
            var el = document.getElementById(id);
            if (el) el.dispatchEvent(new Event('input', {bubbles: true}));
        });

        // Highlight active preset card
        document.querySelectorAll('.preset-card').forEach(function(c) {
            c.style.borderColor = '';
            c.style.boxShadow = '';
        });
        var card = document.getElementById('preset-' + name);
        if (card) {
            card.style.borderColor = '#6366f1';
            card.style.boxShadow = '0 0 0 3px rgba(99,102,241,0.3)';
        }
    }

    function openSavePresetDialog() {
        var name = prompt('输入自定义预设名称：');
        if (!name || !name.trim()) return;
        var vals = getPresetVals();
        var ok = saveCustomPreset(name.trim(), vals);
        if (ok) {
            alert('✅ 预设 "' + name.trim() + '" 已保存！\\n刷新页面后可使用。');
            location.reload();
        } else {
            alert('❌ 保存失败，请重试。');
        }
    }

    // Load custom presets on page load
    document.addEventListener('DOMContentLoaded', function() {
        var custom = loadCustomPresets();
        Object.keys(custom).forEach(function(name) {
            var vals = custom[name];
            // Create a custom preset card dynamically
            var container = document.querySelector('.preset-cards-row');
            if (!container) return;
            var div = document.createElement('div');
            div.className = 'preset-card';
            div.id = 'preset-custom-' + name;
            div.onclick = function() { applyCustomPreset(vals); };
            div.innerHTML = '<span class="preset-icon">⭐</span>'
                + '<span class="preset-label">' + name + '</span>'
                + '<span class="preset-desc">自定义</span>';
            container.appendChild(div);
        });
    });

    function applyCustomPreset(vals) {
        var ids = ['num-colors-slider','min-region-slider','curve-error-slider',
                   'corner-angle-slider','smoothness-slider','smooth-spatial-slider',
                   'smooth-color-slider','upscale-slider','max-pixels-slider','subpixel-checkbox'];
        ids.forEach(function(id, i) {
            var el = document.getElementById(id);
            if (el) {
                if (el.type === 'range') el.value = vals[i];
                else if (el.type === 'checkbox') el.checked = vals[i];
            }
        });
        ids.forEach(function(id) {
            var el = document.getElementById(id);
            if (el) el.dispatchEvent(new Event('input', {bubbles: true}));
        });
    }
    </script>
    """

    def on_vectorize(
        image_file, num_colors, min_region_area, curve_fit_error,
        corner_angle_threshold, smoothness, smoothing_spatial,
        smoothing_color, upscale_short_edge, max_working_pixels,
        enable_subpixel_refine,
    ):
        if image_file is None:
            gr.Info("请先上传一张图片。")
            return (
                [gr.no_update(), "", PALETTE_PLACEHOLDER, "",
                 None, None, CSS_PLACEHOLDER, None]
            )

        file_size = Path(image_file).stat().st_size
        if file_size > 50 * 1024 * 1024:
            gr.Info(f"文件过大（{file_size / 1024 / 1024:.1f} MB）。限制为 50 MB。")
            return (
                [gr.no_update(), "", PALETTE_PLACEHOLDER, "",
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

            return (
                [comparison_value, metadata, palette_html, palette_css,
                 svg_path, png_path, svg_content]
            )

        except Exception as exc:
            gr.Warning(f"矢量化失败：{exc}")
            return (
                [gr.no_update(), f"Error: {exc}", PALETTE_PLACEHOLDER, "",
                 None, None, CSS_PLACEHOLDER, None]
            )

    def on_clear():
        return (
            [None, "", PALETTE_PLACEHOLDER, "",
             None, None, CSS_PLACEHOLDER, None]
        )

    # Main vectorize event
    vectorize_btn.click(
        on_vectorize,
        inputs=[image_input, num_colors, min_region_area, curve_fit_error,
                corner_angle_threshold, smoothness, smoothing_spatial,
                smoothing_color, upscale_short_edge, max_working_pixels,
                enable_subpixel_refine],
        outputs=[comparison_slider, metadata_output, palette_output, palette_export,
                 svg_download, png_download, svg_source_hidden],
        show_progress="full",
    )

    clear_btn.click(
        on_clear,
        outputs=[image_input, metadata_output, palette_output,
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
        head=SVG_PREVIEW_JS + COPY_FEEDBACK_JS + preset_js,
    )
