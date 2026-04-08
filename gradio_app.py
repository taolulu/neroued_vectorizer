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
    """Build interactive palette HTML with usage %."""
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

BLOCK_TITLE = "neroued 矢量化器"
DESCRIPTION = (
    "高质量栅格转 SVG 矢量化工具。上传图片、调整下方的参数，点击 **矢量化** 即可生成 SVG。"
)

PALETTE_PLACEHOLDER = (
    '<div style="color:#aaa;font-size:14px;">调色板将在此处显示</div>'
)

CSS_PLACEHOLDER = (
    '<div style="color:#aaa;font-size:13px;font-family:monospace;">CSS 变量将在此处显示</div>'
)

PRESETS = {
    "Photograph": (0, 0, 1.5, 80.0, 0.2, 8.0, 8.0, 0, 4000000, True),
    "Illustration": (16, 10, 1.0, 60.0, 0.5, 12.0, 12.0, 0, 6000000, True),
    "Line Art": (4, 5, 0.5, 120.0, 0.8, 5.0, 5.0, 0, 2000000, False),
}

# ── Gradio app ────────────────────────────────────────────────────────────────

with gr.Blocks(title=BLOCK_TITLE) as demo:
    gr.Markdown(f"# {BLOCK_TITLE}")
    gr.Markdown(DESCRIPTION)

    # ── Presets row ──────────────────────────────────────────────────────────
    gr.Markdown("### 预设")
    with gr.Row():
        preset_photograph_btn = gr.Button("照片", size="sm")
        preset_illustration_btn = gr.Button("插画", size="sm")
        preset_lineart_btn = gr.Button("线稿", size="sm")

    gr.Markdown("---")

    # ── Main input row ───────────────────────────────────────────────────────
    with gr.Row():
        image_input = gr.Image(
            label="输入图片",
            type="filepath",
            height=400,
        )
        with gr.Row():
            vectorize_btn = gr.Button("矢量化", variant="primary")
            clear_btn = gr.Button("清除")

    # ── Comparison slider ─────────────────────────────────────────────────────
    gr.Markdown("### 效果对比")
    with gr.Row():
        with gr.Column(scale=2):
            comparison_slider = gr.ImageSlider(
                label="拖动滑块对比 — 左侧：原图  |  右侧：矢量化结果",
                slider_position=50,
                height=460,
            )
        with gr.Column(scale=1):
            metadata_output = gr.Textbox(
                label="元数据",
                lines=6,
                interactive=False,
            )

    # ── Parameters ───────────────────────────────────────────────────────────
    gr.Markdown("### 参数配置")

    with gr.Row():
        with gr.Column():
            num_colors = gr.Slider(
                label="颜色数量（0 = 自动）",
                minimum=0, maximum=256, value=0, step=1,
            )
            min_region_area = gr.Slider(
                label="最小区域面积（px²）",
                minimum=0, maximum=500, value=0, step=1,
            )
            curve_fit_error = gr.Slider(
                label="曲线拟合误差（px）",
                minimum=0.1, maximum=10.0, value=1.0, step=0.1,
            )
            corner_angle_threshold = gr.Slider(
                label="拐角角度阈值（°）",
                minimum=1.0, maximum=180.0, value=60.0, step=1.0,
            )
            smoothness = gr.Slider(
                label="平滑度 [0, 1]",
                minimum=0.0, maximum=1.0, value=0.3, step=0.01,
            )

        with gr.Column():
            smoothing_spatial = gr.Slider(
                label="平滑空间半径",
                minimum=0.0, maximum=50.0, value=10.0, step=0.5,
            )
            smoothing_color = gr.Slider(
                label="平滑色彩半径",
                minimum=0.0, maximum=50.0, value=10.0, step=0.5,
            )
            upscale_short_edge = gr.Slider(
                label="放大短边（0 = 关闭）",
                minimum=0, maximum=2000, value=0, step=1,
            )
            max_working_pixels = gr.Slider(
                label="最大处理像素（0 = 关闭）",
                minimum=0, maximum=50000000, value=4000000, step=100000,
            )
            enable_subpixel_refine = gr.Checkbox(
                label="启用亚像素细化",
                value=True,
            )

    # ── Results ─────────────────────────────────────────────────────────────
    gr.Markdown("### 结果")

    # Hidden textbox that stores raw SVG — used by JS copy function
    svg_source_hidden = gr.Textbox(
        label="SVG 源码",
        lines=6,
        visible=False,
        elem_id="svg-source-hidden",
    )

    with gr.Row():
        with gr.Column(scale=1):
            palette_output = gr.HTML(
                value=PALETTE_PLACEHOLDER,
                label="调色板",
            )
            palette_export = gr.Code(
                value="",
                language="css",
                label="调色板（CSS 变量格式）",
                lines=6,
                visible=True,
            )
        with gr.Column(scale=1):
            with gr.Row():
                svg_download = gr.File(label="下载 SVG", visible=True)
                png_download = gr.File(label="下载 PNG（渲染图）", visible=True)
            copy_svg_btn = gr.Button(
                "复制 SVG 源码",
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

    # Main vectorize event with progress
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
            "  if (btn) { btn.textContent = '已复制！'; btn.style.color = '#22c55e'; "
            "    setTimeout(function(){ btn.textContent = '复制 SVG 源码'; btn.style.color = ''; }, 1500); } "
            "}).catch(function(e){ console.error(e); }); "
            "}",
    )


if __name__ == "__main__":
    demo.launch(
        server_port=7861,
        server_name="0.0.0.0",
        head=SVG_PREVIEW_JS + COPY_FEEDBACK_JS,
    )
