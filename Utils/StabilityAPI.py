import base64
import io
import os
import re
from os import listdir
from os.path import isfile, join
from io import BytesIO
import requests
from PIL import Image
import tempfile

class StabilityAPI:
    def __init__(self, api_key):
        self.tmpdir = "cloudinary_images"
        self.engine_id = "stable-diffusion-v1-5"
        self.api_host = os.getenv('API_HOST', 'https://api.stability.ai')
        self.api_key = api_key

        if self.api_key is None:
            raise Exception("Stability API key is required.")

    def text_to_image(self, actor_name, text_prompts, cfg_scale=7, clip_guidance_preset="FAST_BLUE", height=512, width=512, samples=1, steps=30):
        if self.api_key:
            response = requests.post(
                f"{self.api_host}/v1/generation/{self.engine_id}/text-to-image",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "text_prompts": [{"text": text} for text in text_prompts],
                    "cfg_scale": cfg_scale,
                    "clip_guidance_preset": clip_guidance_preset,
                    "height": height,
                    "width": width,
                    "samples": samples,

                    "steps": steps,
                },
            )

            if response.status_code != 200:
                raise Exception("Non-200 response: " + str(response.text))

            data = response.json()
            image_data = base64.b64decode(data["artifacts"][0]["base64"])
            file_name = f"{actor_name}.png"
            img = Image.open(io.BytesIO(image_data))
            img.save(os.path.join(self.tmpdir, file_name), quality=85)
            return os.path.join(self.tmpdir, file_name)

    def image_to_image(self, actor_name, init_image, text_prompts, cfg_scale=7, clip_guidance_preset="FAST_BLUE", samples=1, steps=30):
        if self.api_key:
            response = requests.post(
                f"{self.api_host}/v1/generation/{self.engine_id}/image-to-image",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                files={
                    "init_image": open(init_image, "rb")
                },
                data={
                    "image_strength": 0.35,
                    "init_image_mode": "IMAGE_STRENGTH",
                    "text_prompts": [{"text": text} for text in text_prompts],
                    "cfg_scale": cfg_scale,
                    "clip_guidance_preset": clip_guidance_preset,
                    "samples": samples,
                    "steps": steps,
                }
            )

            if response.status_code != 200:
                raise Exception("Non-200 response: " + str(response.text))

            data = response.json()
            image_data = base64.b64decode(data["artifacts"][0]["base64"])
            file_name = f"{actor_name}.png"
            img = Image.open(io.BytesIO(image_data))
            img.save(os.path.join(self.tmpdir, file_name), quality=85)
            return os.path.join(self.tmpdir, file_name)            