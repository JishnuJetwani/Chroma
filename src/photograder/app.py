"""A local JPEG and PNG grading workbench."""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from .inference import Grader, adjustment_overlay


def _jpeg_tab(gr, grader, lock, export_dir):
    pilot = grader.checkpoint.get("metadata", {}).get("pilot", False)
    status = "Research preview · early trained model" if pilot else "Local photo grading"
    architecture = grader.model.config.architecture
    has_global = architecture not in ("unet", "bilateral")
    has_local = architecture not in ("fixed", "global")

    def maps(analysis, treatment):
        shown = replace(analysis, recipe=grader._recipe(analysis, treatment))
        return adjustment_overlay(shown, "exposure"), adjustment_overlay(shown, "log_contrast")

    def analyze(path, style, global_strength, local_strength, treatment):
        if not path:
            raise gr.Error("Open a JPEG or PNG photo first.")
        try:
            with lock:
                analysis = grader.analyze(path, style)
                result = grader.render_preview(analysis, global_strength, local_strength, treatment)
                brightness, contrast_map = maps(analysis, treatment)
            label = "Backlight recovery · experimental" if treatment == "backlight" else analysis.style
            return analysis, (analysis.preview, result), brightness, contrast_map, f"{label} · ready", None
        except (ValueError, OSError, RuntimeError) as error:
            raise gr.Error(str(error)) from error

    def re_render(analysis, global_strength, local_strength, treatment):
        if analysis is None:
            return None, None, None, None, "Press Auto Grade when you're ready."
        with lock:
            result = grader.render_preview(analysis, global_strength, local_strength, treatment)
            brightness, contrast_map = maps(analysis, treatment)
        label = "Backlight recovery · experimental" if treatment == "backlight" else analysis.style
        return (analysis.preview, result), None, brightness, contrast_map, f"{label} · ready"

    def change_style(path, style, global_strength, local_strength, treatment):
        if not path:
            return None, None, None, None, "Open a photo to try this style.", None
        return analyze(path, style, global_strength, local_strength, treatment)

    def export(analysis, global_strength, local_strength, extension, treatment):
        if analysis is None:
            raise gr.Error("Grade a photo before exporting.")
        filename = Path(analysis.source_path).stem + "-graded-" + uuid.uuid4().hex[:8] + extension
        with lock:
            result = grader.export(
                analysis, Path(export_dir) / filename, global_strength, local_strength, treatment
            )
        return result["path"]

    gr.Markdown(status)
    state = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            photo = gr.Image(
                type="filepath", image_mode=None, sources=["upload"], label="Your photo", height=250
            )
            style = gr.Dropdown(grader.styles, value=grader.styles[0], label="Style")
            treatment = gr.Dropdown(
                [("Standard grade", "model"), ("Backlight recovery (experimental)", "backlight")],
                value="model",
                label="Treatment",
                visible=architecture == "local",
                info="Try Backlight recovery for washed-out photos with strong sunlight. Preserves the glow.",
            )
            button = gr.Button("Auto Grade", variant="primary")
            grade_strength = gr.Slider(0, 1, value=1, step=0.01, label="Overall grade", visible=has_global)
            local_strength = gr.Slider(
                0,
                1,
                value=1,
                step=0.01,
                label="Local correction" if has_global else "Enhancement strength",
                visible=has_local,
            )
            message = gr.Markdown("Your photo stays on this computer.")
        with gr.Column(scale=3):
            comparison = gr.ImageSlider(
                label="Original / graded", type="pil", format="png", interactive=False
            )
    with gr.Accordion(
        "See predicted local adjustments", open=False, visible=architecture in ("local", "cascade")
    ):
        gr.Markdown(
            "Full-strength adjustments: orange shows increases; blue shows decreases. "
            "The strength slider scales these adjustments."
        )
        with gr.Row():
            exposure = gr.Image(label="Brightness", type="pil", format="png", interactive=False)
            contrast = gr.Image(label="Contrast", type="pil", format="png", interactive=False)
    with gr.Row():
        extension = gr.Radio([".png", ".jpg"], value=".png", label="Export format")
        export_button = gr.Button("Export original resolution")
        download = gr.File(label="Graded photo", interactive=False)
    button.click(
        analyze,
        [photo, style, grade_strength, local_strength, treatment],
        [state, comparison, exposure, contrast, message, download],
        concurrency_id="grader",
    )
    # A new photo invalidates the cached recipe; sliders never trigger model analysis.
    photo.change(
        lambda: (None, None, None, None, None, "Press Auto Grade when you're ready."),
        outputs=[state, comparison, exposure, contrast, download, message],
        concurrency_id="grader",
    )
    style.change(
        change_style,
        [photo, style, grade_strength, local_strength, treatment],
        [state, comparison, exposure, contrast, message, download],
        concurrency_id="grader",
    )
    grade_strength.release(
        re_render,
        [state, grade_strength, local_strength, treatment],
        [comparison, download, exposure, contrast, message],
        concurrency_id="grader",
    )
    local_strength.release(
        re_render,
        [state, grade_strength, local_strength, treatment],
        [comparison, download, exposure, contrast, message],
        concurrency_id="grader",
    )
    treatment.change(
        re_render,
        [state, grade_strength, local_strength, treatment],
        [comparison, download, exposure, contrast, message],
        concurrency_id="grader",
    )
    export_button.click(
        export,
        [state, grade_strength, local_strength, extension, treatment],
        [download],
        concurrency_id="grader",
    )


def create_app(checkpoint, device="auto", export_dir="exports"):
    """Open a local JPEG/PNG grading workbench."""
    import gradio as gr

    grader = Grader(checkpoint, device)
    with gr.Blocks(title="Selective Grade", analytics_enabled=False) as demo:
        gr.Markdown("# Selective Grade\nAutomatic color grading with local light adjustments.")
        _jpeg_tab(gr, grader, threading.Lock(), export_dir)
    return demo


def launch(checkpoint, device="auto", host="127.0.0.1", port=7860, export_dir="exports"):
    demo = create_app(checkpoint, device, export_dir)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=host, server_port=port, share=False, allowed_paths=[str(Path(export_dir).resolve())]
    )
