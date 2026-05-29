import subprocess
import sys
import uuid
import re
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "minimal_sd3_run.py"
OUTPUT_ROOT = ROOT / "generated_images" / "gradio_runs"

SUPPORTED_SHAPES = [
    (512, 512),
    (512, 768),
    (768, 512),
    (576, 1024),
    (1024, 576),
    (768, 1024),
    (1024, 768),
    (1024, 1024),
]

SUPPORTED_WIDTHS = sorted({width for width, _ in SUPPORTED_SHAPES})
SUPPORTED_HEIGHTS_BY_WIDTH = {
    width: sorted(height for shape_width, height in SUPPORTED_SHAPES if shape_width == width)
    for width in SUPPORTED_WIDTHS
}

PIPELINE_DEFAULTS = {
    "sd3_base": {
        "num_inference_steps": 30,
        "width": 1024,
        "height": 1024,
        "guidance_scale": 5.0,
        "dynamic_shape": True,
    },
    "sd35_base": {
        "num_inference_steps": 40,
        "width": 1024,
        "height": 1024,
        "guidance_scale": 5.0,
        "dynamic_shape": True,
    },
}


def _optional_str(value):
    value = "" if value is None else str(value).strip()
    return value or None


def _as_int(value, name):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise gr.Error(f"{name} must be an integer")


def _as_float(value, name):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise gr.Error(f"{name} must be a number")


def _parse_saved_images(output_text: str):
    images = []
    marker = "[Image saved]"
    for line in output_text.splitlines():
        if marker in line:
            path_text = line.split(marker, 1)[1].strip()
            if path_text:
                image_path = Path(path_text)
                if not image_path.is_absolute():
                    image_path = ROOT / image_path
                if image_path.exists():
                    images.append(str(image_path))
    return images


def generate_image(
    pipeline,
    prompt,
    width,
    height,
    num_inference_steps,
    guidance_scale,
    seed,
    negative_prompt,
    num_images_per_prompt,
    dynamic_shape,
    controlnet,
    model_id,
    revision,
    model_path,
    custom_op_path,
    progress=gr.Progress(track_tqdm=True),
):
    prompt = _optional_str(prompt)
    if not prompt:
        raise gr.Error("Prompt is required")

    width = _as_int(width, "Width")
    height = _as_int(height, "Height")
    num_inference_steps = _as_int(num_inference_steps, "Inference steps")
    num_images_per_prompt = _as_int(num_images_per_prompt, "Images per prompt")
    guidance_scale = _as_float(guidance_scale, "Guidance scale")

    if dynamic_shape and (width, height) not in SUPPORTED_SHAPES:
        supported = ", ".join(f"{w}x{h}" for w, h in SUPPORTED_SHAPES)
        raise gr.Error(f"Unsupported dynamic shape {width}x{height}. Supported: {supported}")

    run_dir = OUTPUT_ROOT / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(RUNNER),
        "--pipelines",
        pipeline,
        "--prompt",
        prompt,
        "--width",
        str(width),
        "--height",
        str(height),
        "--num_inference_steps",
        str(num_inference_steps),
        "--guidance_scale",
        str(guidance_scale),
        "--num_images_per_prompt",
        str(num_images_per_prompt),
        "--controlnet",
        _optional_str(controlnet) or "None",
        "--output_path",
        str(run_dir),
    ]

    if dynamic_shape:
        cmd.append("--dynamic_shape")
    else:
        cmd.append("--no-dynamic_shape")

    seed = _optional_str(seed)
    if seed is not None:
        cmd.extend(["--seed", str(_as_int(seed, "Seed"))])

    optional_args = {
        "--n_prompt": negative_prompt,
        "--model_id": model_id,
        "--revision": revision,
        "--model_path": model_path,
        "--custom_op_path": custom_op_path,
    }
    for arg_name, arg_value in optional_args.items():
        arg_value = _optional_str(arg_value)
        if arg_value is not None:
            cmd.extend([arg_name, arg_value])

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
    except Exception as exc:
        raise gr.Error(f"Failed to start generation: {exc}") from exc

    log_lines = [f"Command:\n{' '.join(cmd)}\n"]
    progress(0, desc="Loading models...")

    if process.stdout is not None:
        for line in process.stdout:
            log_lines.append(line.rstrip())
            match = re.search(r"__STEP_COMPLETE__\s+(\d+)/(\d+)", line)
            if match:
                current_step = int(match.group(1))
                total_steps = int(match.group(2))
                progress(current_step / total_steps, desc=f"Denoising {current_step}/{total_steps}")

    return_code = process.wait()
    log_text = "\n".join(log_lines)

    if return_code != 0:
        raise gr.Error(f"Generation failed. See log output.\n\n{log_text[-4000:]}")

    progress(1.0, desc="Finalizing image...")
    images = _parse_saved_images(log_text)
    if not images:
        images = [str(path) for path in sorted(run_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)]

    if not images:
        raise gr.Error(f"Generation finished but no image was found.\n\n{log_text[-4000:]}")

    return images[0], images, log_text


def apply_pipeline_defaults(pipeline):
    defaults = PIPELINE_DEFAULTS[pipeline]
    return (
        defaults["width"],
        gr.update(choices=SUPPORTED_HEIGHTS_BY_WIDTH[defaults["width"]], value=defaults["height"]),
        defaults["num_inference_steps"],
        defaults["guidance_scale"],
        defaults["dynamic_shape"],
    )


def update_height_options(selected_width, current_height):
    selected_width = _as_int(selected_width, "Width")
    allowed_heights = SUPPORTED_HEIGHTS_BY_WIDTH[selected_width]
    current_height = _as_int(current_height, "Height") if current_height is not None else None
    return gr.update(
        choices=allowed_heights,
        value=current_height if current_height in allowed_heights else allowed_heights[0],
    )


with gr.Blocks(title="Minimal SD3 AMD NPU Generator") as demo:
    gr.Markdown("# Minimal SD3 / SD3.5 Generator")
    gr.Markdown("Runs `minimal_sd3_run.py` and displays the generated image.")

    with gr.Row():
        with gr.Column(scale=1):
            pipeline = gr.Dropdown(
                choices=list(PIPELINE_DEFAULTS),
                value="sd35_base",
                label="Pipeline",
            )
            prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Enter prompt...")
            negative_prompt = gr.Textbox(label="Negative prompt", value="", lines=2)

            with gr.Row():
                width = gr.Dropdown(label="Width", choices=SUPPORTED_WIDTHS, value=1024)
                height = gr.Dropdown(label="Height", choices=SUPPORTED_HEIGHTS_BY_WIDTH[1024], value=1024)

            with gr.Row():
                num_inference_steps = gr.Number(label="Inference steps", value=40, precision=0)
                guidance_scale = gr.Number(label="Guidance scale", value=5.0)

            with gr.Row():
                seed = gr.Textbox(label="Seed (blank = random)", value="")
                num_images_per_prompt = gr.Number(label="Images per prompt", value=1, precision=0)

            dynamic_shape = gr.Checkbox(label="Dynamic shape", value=True)
            controlnet = gr.Textbox(label="ControlNet", value="None")

            with gr.Accordion("Advanced", open=False):
                model_id = gr.Textbox(label="Override model_id", value="")
                revision = gr.Textbox(label="Revision", value="")
                model_path = gr.Textbox(label="Model path", value="")
                custom_op_path = gr.Textbox(
                    label="Custom op path",
                    value="C:/Program Files/RyzenAI/1.7.1/deployment/onnx_custom_ops.dll",
                )

            generate_button = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            image_output = gr.Image(label="Generated image", type="filepath")
            gallery_output = gr.Gallery(label="All generated images", columns=2, height="auto")
            log_output = gr.Textbox(label="Log", lines=18)

    pipeline.change(
        apply_pipeline_defaults,
        inputs=pipeline,
        outputs=[width, height, num_inference_steps, guidance_scale, dynamic_shape],
    )

    width.change(
        update_height_options,
        inputs=[width, height],
        outputs=height,
    )

    generate_button.click(
        generate_image,
        inputs=[
            pipeline,
            prompt,
            width,
            height,
            num_inference_steps,
            guidance_scale,
            seed,
            negative_prompt,
            num_images_per_prompt,
            dynamic_shape,
            controlnet,
            model_id,
            revision,
            model_path,
            custom_op_path,
        ],
        outputs=[image_output, gallery_output, log_output],
    )


if __name__ == "__main__":
    demo.launch()
