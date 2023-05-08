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


import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

class DiffusionImageGenerator:
    def __init__(self, model_id="stabilityai/stable-diffusion-2", scheduler_subfolder="scheduler"):
        self.tmpdir = "cloudinary_images"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.scheduler_subfolder = scheduler_subfolder
        self.pipe = None
        self.fallback_image = "https://plchldr.co/i/336x280"

    def initialize_pipe(self):
        scheduler = EulerDiscreteScheduler.from_pretrained(self.model_id, subfolder=self.scheduler_subfolder)
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, scheduler=scheduler)
        self.pipe = self.pipe.to(self.device)

    def generate_image(self, prompt, actorname):
        if not self.pipe:
            self.initialize_pipe()

        image = self.pipe(prompt).images[0]
        file_name = f"{actorname}.png"
        image.save(os.path.join(self.tmpdir, file_name))
        return os.path.join(self.tmpdir, file_name)

