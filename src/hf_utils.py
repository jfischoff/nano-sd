import os

from huggingface_hub import create_repo, upload_folder
from PIL import Image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def save_model_card(repo_id: str, image_logs=None, base_model=str, folder_path=None):
    img_str = ""
    if image_logs is not None:
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]

            img_str += f"prompt: {validation_prompt}\n"
            # images = images
            image_grid(images, 1, len(images)).save(os.path.join(folder_path, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- jax-diffusers-event
inference: true
---
    """
    model_card = f"""
# controlnet- {repo_id}

These are unet weights trained scratch but using {base_model} for everything else. \n
{img_str}
"""
    with open(os.path.join(folder_path, "README.md"), "w") as f:
        f.write(yaml + model_card)


def publish_to_hub(repo_id: str, token, image_logs=None, base_model=str, folder_path=None):
    repo_id = create_repo(repo_id, exist_ok=True, token=token).repo_id
    save_model_card(repo_id, image_logs, base_model, folder_path)
    upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        commit_message="End of training",
        ignore_patterns=["step_*", "epoch_*"],
    )
