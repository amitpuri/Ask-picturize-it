import io
import os
import re
from os import listdir
from os.path import isfile, join
from io import BytesIO
import requests
from PIL import Image
import tempfile

class ImageUtils:
    def __init__(self):
        self.tmpdir = "cloudinary_images"
        self.fallback_image = "https://plchldr.co/i/336x280"

    def url_to_image(self, url):
        if len(url)==0:
            return None
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        else:
            response = requests.get(self.fallback_image)
            img = Image.open(BytesIO(response.content))
            return img


    def parse_image_name(self, response):
        left_identifier = "img-"
        right_identifier = ".png"
        file_name_index = 6
        pictures_list = [self.write_image(data["url"], f"{left_identifier}{re.findall('{}(.*){}'.format(left_identifier, right_identifier), data['url'].split('/')[file_name_index])[0]}{right_identifier}") for data in response]
        return pictures_list


    def fallback_image_array_implement(self):
        pictures_list = [self.write_image(self.fallback_image, "fallback_image.png")]
        return pictures_list

    
    def write_image(self, image_url, file_name):
        buffer = tempfile.SpooledTemporaryFile(max_size=1e9)
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=1024):
                buffer.write(chunk)
            buffer.seek(0)
            img = Image.open(io.BytesIO(buffer.read()))
        else:
            response = requests.get(self.fallback_image)
            img = Image.open(BytesIO(response.content))
        
        img.save(os.path.join(self.tmpdir, file_name), quality=85)
        buffer.close()
        return os.path.join(self.tmpdir, file_name)
           

    def fallback_image_implement(self):
        return self.url_to_image(self.fallback_image)
