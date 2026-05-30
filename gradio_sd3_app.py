import io
import os
import secrets
import threading
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "generated_images" / "gradio_runs"
DEFAULT_CUSTOM_OP_PATH = "C:/Program Files/RyzenAI/1.7.1/deployment/onnx_custom_ops.dll"

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
        "model_id": "stabilityai/stable-diffusion-3-medium-amdnpu",
        "num_inference_steps": 30,
        "width": 1024,
        "height": 1024,
        "guidance_scale": 5.0,
        "dynamic_shape": True,
    },
    "sd35_base": {
        "model_id": "stabilityai/stable-diffusion-3.5-medium-amdnpu",
        "num_inference_steps": 40,
        "width": 1024,
        "height": 1024,
        "guidance_scale": 5.0,
        "dynamic_shape": True,
    },
}

_LOADED_MODEL_KEY = None
_LOADED_PIPE_TRIGGER = None
_MODEL_LOCK = threading.Lock()


def pick(value, default):
    return default if value is None else value


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


def _normalize_controlnet(controlnet):
    return _optional_str(controlnet) or "None"


def _custom_op_path(custom_op_path):
    return _optional_str(custom_op_path) or DEFAULT_CUSTOM_OP_PATH


def _load_pipeline_dependencies():
    from src.StableDiffusion3PipelineTrigger import StableDiffusion3PipelineTrigger
    from src.utils import common

    return StableDiffusion3PipelineTrigger, common


def _release_loaded_model():
    global _LOADED_MODEL_KEY, _LOADED_PIPE_TRIGGER

    if _LOADED_PIPE_TRIGGER is not None:
        _LOADED_PIPE_TRIGGER.__exit__(None, None, None)
    _LOADED_MODEL_KEY = None
    _LOADED_PIPE_TRIGGER = None


def _model_cache_key(pipeline, width, dynamic_shape, controlnet, model_id, revision, model_path, custom_op_path):
    defaults = PIPELINE_DEFAULTS[pipeline]
    resolved_model_id = pick(_optional_str(model_id), defaults["model_id"])
    return (
        pipeline,
        resolved_model_id,
        _optional_str(revision),
        _optional_str(model_path),
        _custom_op_path(custom_op_path),
        _normalize_controlnet(controlnet),
        bool(dynamic_shape),
        None if dynamic_shape else _as_int(width, "Width"),
    )


def _get_pipe_trigger(pipeline, width, dynamic_shape, controlnet, model_id, revision, model_path, custom_op_path):
    global _LOADED_MODEL_KEY, _LOADED_PIPE_TRIGGER

    key = _model_cache_key(
        pipeline,
        width,
        dynamic_shape,
        controlnet,
        model_id,
        revision,
        model_path,
        custom_op_path,
    )
    if _LOADED_PIPE_TRIGGER is not None and _LOADED_MODEL_KEY == key:
        return _LOADED_PIPE_TRIGGER, False

    _release_loaded_model()

    defaults = PIPELINE_DEFAULTS[pipeline]
    project_root = ROOT
    os.environ["DD_PLUGINS_ROOT"] = str((project_root / "lib" / "transaction" / "stx").resolve())
    os.environ["DD_ROOT"] = str((project_root / "lib").resolve())

    if not Path(os.environ["DD_PLUGINS_ROOT"]).exists():
        raise FileNotFoundError(f"DD_PLUGINS_ROOT not found: {os.environ['DD_PLUGINS_ROOT']}")

    if not Path(os.environ["DD_ROOT"]).exists():
        raise FileNotFoundError(f"DD_ROOT not found: {os.environ['DD_ROOT']}")

    StableDiffusion3PipelineTrigger, _ = _load_pipeline_dependencies()
    pipe_trigger = StableDiffusion3PipelineTrigger(
        model_id=pick(_optional_str(model_id), defaults["model_id"]),
        custom_op_path=_custom_op_path(custom_op_path),
        root_path=".",
        model_path=_optional_str(model_path),
        sub_model_path="normal",
        common_model_path="common",
        controlnet_str=_normalize_controlnet(controlnet),
        enable_compile=False,
        enable_profile=False,
        profiling_rounds=1,
        width=_as_int(width, "Width"),
        t5_sequence_len=83,
        is_dynamic=bool(dynamic_shape),
        revision=_optional_str(revision),
    )
    _LOADED_PIPE_TRIGGER = pipe_trigger
    _LOADED_MODEL_KEY = key
    return pipe_trigger, True


def preload_model(pipeline, width, dynamic_shape, controlnet, model_id, revision, model_path, custom_op_path):
    try:
        with _MODEL_LOCK:
            _, was_loaded = _get_pipe_trigger(
                pipeline,
                width,
                dynamic_shape,
                controlnet,
                model_id,
                revision,
                model_path,
                custom_op_path,
            )
    except Exception as exc:
        raise gr.Error(f"Failed to load model: {exc}") from exc

    action = "Loaded" if was_loaded else "Reusing"
    return f"{action} model for {pipeline}."


def preload_default_model():
    defaults = PIPELINE_DEFAULTS["sd35_base"]
    return preload_model(
        "sd35_base",
        defaults["width"],
        defaults["dynamic_shape"],
        "None",
        "",
        "",
        "",
        DEFAULT_CUSTOM_OP_PATH,
    )


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

    seed = _optional_str(seed)
    if seed is None:
        seed = secrets.randbits(32)
        seed_line = f"Random seed = {seed}"
    else:
        seed = _as_int(seed, "Seed")
        seed_line = f"Seed = {seed}"

    defaults = PIPELINE_DEFAULTS[pipeline]
    resolved_model_id = pick(_optional_str(model_id), defaults["model_id"])
    controlnet = _normalize_controlnet(controlnet)
    negative_prompt = _optional_str(negative_prompt)

    progress(0, desc="Loading models...")
    log_buffer = io.StringIO()
    log_lines = [f"Pipeline: {pipeline}", f"Model ID: {resolved_model_id}", seed_line, f"Output path: {run_dir}"]

    try:
        with _MODEL_LOCK:
            pipe_trigger, was_loaded = _get_pipe_trigger(
                pipeline,
                width,
                dynamic_shape,
                controlnet,
                model_id,
                revision,
                model_path,
                custom_op_path,
            )
            log_lines.extend(
                [
                    f"Model cache: {'loaded' if was_loaded else 'reused'}",
                    f"DD_PLUGINS_ROOT = {os.environ['DD_PLUGINS_ROOT']}",
                    f"DD_ROOT = {os.environ['DD_ROOT']}",
                ]
            )
            with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
                images = pipe_trigger.run(
                    height=height,
                    width=width,
                    prompt=prompt,
                    n_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    control_image_path=None,
                    controlnet_conditioning_scale=None,
                    num_images_per_prompt=num_images_per_prompt,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )
    except Exception as exc:
        captured_log = log_buffer.getvalue().strip()
        log_text = "\n".join(log_lines + ([captured_log] if captured_log else []))
        raise gr.Error(f"Generation failed. See log output.\n\n{log_text[-4000:]}\n\n{exc}") from exc

    progress(1.0, desc="Finalizing image...")
    _, common = _load_pipeline_dependencies()
    saved_images = []
    for image_idx, image in enumerate(images):
        filename = common.generate_filename(
            resolved_model_id,
            width,
            height,
            num_inference_steps,
            prompt_idx=0,
            image_idx=image_idx,
            controlnet=controlnet,
            run_mode="batch",
            suffix=".png",
        )

        image_path = run_dir / filename
        image.save(image_path)
        saved_images.append(str(image_path))
        log_lines.append(f"[Image saved] {image_path}")

    captured_log = log_buffer.getvalue().strip()
    if captured_log:
        log_lines.append(captured_log)
    log_text = "\n".join(log_lines)

    if not saved_images:
        raise gr.Error(f"Generation finished but no image was found.\n\n{log_text[-4000:]}")

    return saved_images[0], saved_images, log_text


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
    gr.Markdown("Loads the selected model once, reuses it for generation, and displays the generated image.")

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
                    value=DEFAULT_CUSTOM_OP_PATH,
                )

            generate_button = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            image_output = gr.Image(label="Generated image", type="filepath")
            gallery_output = gr.Gallery(label="All generated images", columns=2, height="auto")
            log_output = gr.Textbox(label="Log", lines=18)

    model_inputs = [
        pipeline,
        width,
        dynamic_shape,
        controlnet,
        model_id,
        revision,
        model_path,
        custom_op_path,
    ]

    demo.load(
        preload_model,
        inputs=model_inputs,
        outputs=log_output,
    )

    pipeline_change = pipeline.change(
        apply_pipeline_defaults,
        inputs=pipeline,
        outputs=[width, height, num_inference_steps, guidance_scale, dynamic_shape],
    )
    pipeline_change.then(
        preload_model,
        inputs=model_inputs,
        outputs=log_output,
    )

    width_change = width.change(
        update_height_options,
        inputs=[width, height],
        outputs=height,
    )
    width_change.then(
        preload_model,
        inputs=model_inputs,
        outputs=log_output,
    )

    for model_input in [dynamic_shape, controlnet, model_id, revision, model_path, custom_op_path]:
        model_input.change(
            preload_model,
            inputs=model_inputs,
            outputs=log_output,
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
    print(preload_default_model(), flush=True)
    demo.launch()
