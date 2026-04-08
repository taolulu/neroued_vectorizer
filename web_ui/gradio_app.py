"""Gradio Web UI for neroued-vectorizer service."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np

import neroued_vectorizer as nv


def vectorize_image(
    image_file: str | bytes | np.ndarray,
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
) -> tuple[str, str]:
    """Vectorize an image and return SVG content and palette info."""
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

    # Build palette info string
    palette_lines = ["Palette:"]
    for i, color in enumerate(result.palette):
        r, g, b = color.r, color.g, color.b
        r255 = int(round(r * 255))
        g255 = int(round(g * 255))
        b255 = int(round(b * 255))
        hex_color = f"#{r255:02x}{g255:02x}{b255:02x}"
        palette_lines.append(f"  {i + 1}. {hex_color}  rgb({r255},{g255},{b255})")

    metadata = (
        f"Width: {result.width} px\n"
        f"Height: {result.height} px\n"
        f"Shapes: {result.num_shapes}\n"
        f"Colors: {result.resolved_num_colors}\n"
        + "\n".join(palette_lines)
    )

    return result.svg_content, metadata


def save_svg(svg_content: str) -> bytes:
    """Return SVG content as bytes for file download."""
    return svg_content.encode("utf-8")


# ── Gradio UI ────────────────────────────────────────────────────────────────

BLOCK_TITLE = "neroued-vectorizer"
DESCRIPTION = (
    "High-quality raster-to-SVG vectorizer. Upload an image and tune the "
    "parameters below, then click **Vectorize** to generate the SVG."
)

with gr.Blocks(title=BLOCK_TITLE) as demo:
    gr.Markdown(f"# {BLOCK_TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            image_input = gr.Image(
                label="Input Image",
                type="filepath",
                height=400,
            )
            with gr.Row():
                vectorize_btn = gr.Button("Vectorize", variant="primary")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            svg_output = gr.Textbox(
                label="SVG Output (preview)",
                lines=20,
                max_lines=30,
            )

    gr.Markdown("---")
    gr.Markdown("### Configuration")

    with gr.Row():
        with gr.Column():
            num_colors = gr.Slider(
                label="Number of Colors (0 = auto)",
                minimum=0,
                maximum=256,
                value=0,
                step=1,
            )
            min_region_area = gr.Slider(
                label="Min Region Area (px²)",
                minimum=0,
                maximum=500,
                value=0,
                step=1,
            )
            curve_fit_error = gr.Slider(
                label="Curve Fit Error (px)",
                minimum=0.1,
                maximum=10.0,
                value=1.0,
                step=0.1,
            )
            corner_angle_threshold = gr.Slider(
                label="Corner Angle Threshold (°)",
                minimum=1.0,
                maximum=180.0,
                value=60.0,
                step=1.0,
            )
            smoothness = gr.Slider(
                label="Smoothness [0, 1]",
                minimum=0.0,
                maximum=1.0,
                value=0.3,
                step=0.01,
            )

        with gr.Column():
            smoothing_spatial = gr.Slider(
                label="Smoothing Spatial Radius",
                minimum=0.0,
                maximum=50.0,
                value=10.0,
                step=0.5,
            )
            smoothing_color = gr.Slider(
                label="Smoothing Color Radius",
                minimum=0.0,
                maximum=50.0,
                value=10.0,
                step=0.5,
            )
            upscale_short_edge = gr.Slider(
                label="Upscale Short Edge (0 = off)",
                minimum=0,
                maximum=2000,
                value=0,
                step=1,
            )
            max_working_pixels = gr.Slider(
                label="Max Working Pixels (0 = off)",
                minimum=0,
                maximum=50000000,
                value=4000000,
                step=100000,
            )
            enable_subpixel_refine = gr.Checkbox(
                label="Enable Sub-pixel Refinement",
                value=True,
            )

    gr.Markdown("---")
    gr.Markdown("### Result")

    with gr.Row():
        metadata_output = gr.Textbox(
            label="Metadata",
            lines=8,
            max_lines=12,
        )
        svg_download = gr.File(
            label="Download SVG",
            visible=True,
        )

    # ── Event handlers ────────────────────────────────────────────────────────

    def on_vectorize(image_file, num_colors, min_region_area, curve_fit_error,
                     corner_angle_threshold, smoothness, smoothing_spatial,
                     smoothing_color, upscale_short_edge, max_working_pixels,
                     enable_subpixel_refine):
        if image_file is None:
            gr.Warning("Please upload an image first.")
            return "", ""

        try:
            svg_content, metadata = vectorize_image(
                image_file,
                num_colors,
                min_region_area,
                curve_fit_error,
                corner_angle_threshold,
                smoothness,
                smoothing_spatial,
                smoothing_color,
                upscale_short_edge,
                max_working_pixels,
                enable_subpixel_refine,
            )
            # Return (svg_text, metadata, svg_file_bytes)
            svg_bytes = svg_content.encode("utf-8")
            return svg_content, metadata, io.BytesIO(svg_bytes)
        except Exception as exc:
            gr.Warning(f"Vectorization failed: {exc}")
            return "", f"Error: {exc}", None

    def on_clear():
        return None, "", "", None

    vectorize_btn.click(
        on_vectorize,
        inputs=[
            image_input, num_colors, min_region_area, curve_fit_error,
            corner_angle_threshold, smoothness, smoothing_spatial,
            smoothing_color, upscale_short_edge, max_working_pixels,
            enable_subpixel_refine,
        ],
        outputs=[svg_output, metadata_output, svg_download],
    )

    clear_btn.click(
        on_clear,
        inputs=[],
        outputs=[image_input, svg_output, metadata_output, svg_download],
    )


if __name__ == "__main__":
    demo.launch(server_port=7861, server_name="0.0.0.0")
