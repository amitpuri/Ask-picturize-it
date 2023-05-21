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
from torchvision import transforms
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionImageVariationPipeline

class DiffusionImageGenerator:
    def __init__(self, scheduler_subfolder="scheduler"):
        self.tmpdir = "cloudinary_images"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        self.scheduler_subfolder = scheduler_subfolder
        self.pipe = None

    def initialize_pipe_stable_diffusion(self):
        self.model_id="stabilityai/stable-diffusion-2-1"
        scheduler = EulerDiscreteScheduler.from_pretrained(self.model_id, subfolder=self.scheduler_subfolder)
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, scheduler=scheduler)
        
    
    def generate_image(self, prompt, actorname):
        if not self.pipe:
            self.initialize_pipe_stable_diffusion()

        image = self.pipe(prompt).images[0]
        file_name = f"{actorname}.png"
        image.save(os.path.join(self.tmpdir, file_name))
        return os.path.join(self.tmpdir, file_name)

    def image_variation(self, input_image, actorname="actorname", n_samples=1):
        self.pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            )
        self.pipe = self.pipe.to(self.device)
        scale=3.0
        steps=25
        seed=0
        
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),

            transforms.Normalize(
              [0.48145466, 0.4578275, 0.40821073],
              [0.26862954, 0.26130258, 0.27577711]),
        ])
        
        print(input_image)       
        try:
            inp = tform(Image.open(input_image)).to(self.device)        
        except Exception as exception:
            print(f"Exception Name: {type(exception).__name__}")
            print(exception)
            return "https://plchldr.co/i/336x280"
        
        images_list = self.pipe(
            inp.tile(n_samples, 1, 1, 1),
            guidance_scale=scale,
            num_inference_steps=steps,
            generator=generator,
        )
  
        image = images_list["images"]["nsfw_content_detected"][0]
        file_name = f"{actorname}.png"
        image.save(os.path.join(self.tmpdir, file_name))
        return os.path.join(self.tmpdir, file_name)
