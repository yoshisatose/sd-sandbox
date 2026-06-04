from pathlib import Path
import shutil
import gradio_sd3_app as sdapp


def generate_from_notebook(
    prompt,
    model_name="sd35_base",
    output_file="output.png",
    width=None,
    height=None,
    steps=None,
    guidance_scale=None,
    seed=None,
    negative_prompt="",
    image_count=1,
    overwrite=False,
):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if model_name in sdapp.PIPELINE_DEFAULTS:
        pipeline = model_name
        model_id = ""
    else:
        pipeline = "sd35_base"
        model_id = model_name

    defaults = sdapp.PIPELINE_DEFAULTS[pipeline]

    old_output_root = sdapp.OUTPUT_ROOT
    sdapp.OUTPUT_ROOT = output_file.parent

    try:
        final_result = None

        for status_html, image_paths, log_text in sdapp.generate_image(
            pipeline=pipeline,
            prompt=prompt,
            width=width or defaults["width"],
            height=height or defaults["height"],
            num_inference_steps=steps or defaults["num_inference_steps"],
            guidance_scale=guidance_scale or defaults["guidance_scale"],
            seed="" if seed is None else str(seed),
            negative_prompt=negative_prompt,
            image_count=image_count,
            dynamic_shape=defaults["dynamic_shape"],
            controlnet="None",
            model_id=model_id,
            revision="",
            model_path="",
            custom_op_path=sdapp.DEFAULT_CUSTOM_OP_PATH,
        ):
            final_result = (status_html, image_paths, log_text)

        saved_paths = []

        for idx, generated_path in enumerate(final_result[1]):
            generated_path = Path(generated_path)

            if image_count == 1:
                target_path = output_file
            else:
                target_path = output_file.with_name(
                    f"{output_file.stem}_{idx}{output_file.suffix}"
                )

            if target_path.exists() and not overwrite:
                raise FileExistsError(f"File already exists: {target_path}")

            shutil.move(str(generated_path), str(target_path))
            saved_paths.append(str(target_path))

        return {
            "image_paths": saved_paths,
            "log": final_result[2],
        }

    finally:
        sdapp.OUTPUT_ROOT = old_output_root