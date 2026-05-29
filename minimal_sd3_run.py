import argparse
import os
import sys
from pathlib import Path
import secrets

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.StableDiffusion3PipelineTrigger import StableDiffusion3PipelineTrigger
from src.utils import common


PIPELINES = {
    "sd3_base": {
        "model_id": "stabilityai/stable-diffusion-3-medium-amdnpu",
        "num_inference_steps": 30,
        "width": 1024,
        "height": 1024,
        "controlnet": "None",
        "guidance_scale": 5.0,
        "dynamic_shape": True,
    },
    "sd35_base": {
        "model_id": "stabilityai/stable-diffusion-3.5-medium-amdnpu",
        "num_inference_steps": 40,
        "width": 1024,
        "height": 1024,
        "controlnet": "None",
        "guidance_scale": 5.0,
        "dynamic_shape": True,
    },
}


def pick(value, default):
    return default if value is None else value


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pipelines",
        required=True,
        choices=PIPELINES.keys(),
        help="Pipeline name: sd3_base or sd35_base",
    )

    parser.add_argument("--prompt", required=True)

    # Configurable values from pipeline_configs.yaml
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--controlnet", default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)

    # Configurable like seed, but defaults to each pipeline's YAML behavior
    parser.add_argument(
        "--dynamic_shape",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use --dynamic_shape or --no-dynamic_shape",
    )

    # Extra direct runtime args
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_prompt", default=None)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--sub_model_path", default="normal")
    parser.add_argument("--common_model_path", default="common")
    parser.add_argument("--root_path", default=".")
    parser.add_argument("--custom_op_path", default="C:/Program Files/RyzenAI/1.7.1/deployment/onnx_custom_ops.dll")
    parser.add_argument("--output_path", default="generated_images")

    args = parser.parse_args()
    defaults = PIPELINES[args.pipelines]

    model_id = pick(args.model_id, defaults["model_id"])
    num_inference_steps = pick(args.num_inference_steps, defaults["num_inference_steps"])
    width = pick(args.width, defaults["width"])
    height = pick(args.height, defaults["height"])
    controlnet = pick(args.controlnet, defaults["controlnet"])
    guidance_scale = pick(args.guidance_scale, defaults["guidance_scale"])
    dynamic_shape = pick(args.dynamic_shape, defaults["dynamic_shape"])

    seed = args.seed
    if seed is None:
        seed = secrets.randbits(32)
        print(f"Random seed = {seed}")
    else:
        print(f"Seed = {seed}")

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    project_root = ROOT

    os.environ["DD_PLUGINS_ROOT"] = str((project_root / "lib" / "transaction" / "stx").resolve())
    os.environ["DD_ROOT"] = str((project_root / "lib").resolve())

    print("DD_PLUGINS_ROOT =", os.environ["DD_PLUGINS_ROOT"])
    print("DD_ROOT =", os.environ["DD_ROOT"])

    if not Path(os.environ["DD_PLUGINS_ROOT"]).exists():
        raise FileNotFoundError(f"DD_PLUGINS_ROOT not found: {os.environ['DD_PLUGINS_ROOT']}")

    if not Path(os.environ["DD_ROOT"]).exists():
        raise FileNotFoundError(f"DD_ROOT not found: {os.environ['DD_ROOT']}")

    pipe_trigger = StableDiffusion3PipelineTrigger(
        model_id=model_id,
        custom_op_path=args.custom_op_path,
        root_path=args.root_path,
        model_path=args.model_path,
        sub_model_path=args.sub_model_path,
        common_model_path=args.common_model_path,
        controlnet_str=controlnet,
        enable_compile=False,
        enable_profile=False,
        profiling_rounds=1,
        width=width,
        t5_sequence_len=83,
        is_dynamic=dynamic_shape,
        revision=args.revision,
    )

    images = pipe_trigger.run(
        height=height,
        width=width,
        prompt=args.prompt,
        n_prompt=args.n_prompt,
        num_inference_steps=num_inference_steps,
        control_image_path=None,
        controlnet_conditioning_scale=None,
        num_images_per_prompt=args.num_images_per_prompt,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    for image_idx, image in enumerate(images):
        filename = common.generate_filename(
            model_id,
            width,
            height,
            num_inference_steps,
            prompt_idx=0,
            image_idx=image_idx,
            controlnet=controlnet,
            run_mode="batch",
            suffix=".png",
        )

        image_path = output_dir / filename
        image.save(image_path)
        print(f"[Image saved] {image_path}")


if __name__ == "__main__":
    main()