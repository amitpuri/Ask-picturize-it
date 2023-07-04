import os
from os.path import join
from diffusers import DiffusionPipeline
import torch

class RunwaymlImageGenerator:
    def __init__(self):
        self.tmpdir = "cloudinary_images"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        self.pipe = None

    def initialize_pipe(self):
        self.model_id="runwayml/stable-diffusion-v1-5"        
        self.pipe = DiffusionPipeline.from_pretrained(self.model_id)
        
    def generate_image(self, prompt):
        if not self.pipe:
            self.initialize_pipe()

        image = self.pipe(prompt).images[0]
        file_name = "Generated-Image.png"
        image.save(os.path.join(self.tmpdir, file_name))
        return os.path.join(self.tmpdir, file_name)
