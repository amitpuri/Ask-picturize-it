from OpenAIUtil.Operations import *
import openai
from Utils.ImageUtils import *

class ImageOperations(Operations):
    def __init__(self, api_key: str, org_id: str):
        if org_id is not None:
            openai.organization = org_id
        openai.api_key = api_key
        self.image_utils = ImageUtils()

    def create_edit_masked_image(self, picture_file: str, mask_image_file: str, prompt: str, imagesize: str, num_images: int):
        if not imagesize:  # defaulting to this image size
            imagesize = "256x256"
        try:
            if openai.api_key is not None:
                response = openai.Image.create_edit(
                              image=open(picture_file, "rb"),
                              mask=open(mask_image_file, "rb"),
                              prompt=prompt,
                              n=num_images,
                              size=imagesize
                )
                image_url = response['data'][0]['url']
            else:
                return label_image_masked_output, self.image_utils.url_to_image(image_url), self.image_utils.parse_image_name(response['data'])
        except openai.error.OpenAIError as error_except:
            print(f"ImageOperations create_edit_masked_image")
            print(error_except.http_status)
            print(error_except.error)
            return label_image_masked_output, self.image_utils.url_to_image(image_url), ""
            

    
    def create_variation_from_image(self, picture_file: str, imagesize: str, num_images: int):
        label_inference_variation = "Switch to Output tab to review it"
        if not picture_file and openai.api_key is None:
            return label_inference_variation, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
        if not imagesize:  # defaulting to this image size
            imagesize = "256x256"
        try:
            with open(picture_file, "rb") as file_path:
                image = Image.open(file_path)
                width, height = 256, 256
                image = image.resize((width, height))
                byte_stream = BytesIO()
                image.save(byte_stream, format='PNG')
                image_byte_array = byte_stream.getvalue()
        except Exception as err:
            print(f"ImageOperations create_variation_from_image {err}")
        try:
            response = openai.Image.create_variation(
                image=image_byte_array,
                n=num_images,
                size=imagesize
            )    
            image_url = response['data'][0]['url']
            return label_inference_variation, self.image_utils.url_to_image(image_url), self.image_utils.parse_image_name(response['data'])
        except openai.error.OpenAIError as error_except:
            print(f"ImageOperations create_variation_from_image")
            print(error_except.http_status)
            print(error_except.error)
            return label_inference_variation, self.image_utils.url_to_image(image_url), ""

            
    
    def create_image_from_prompt(self, prompt: str, imagesize: str, num_images: int):
        label_picturize_it = "Switch to Output tab to review it"
        if not prompt and openai.api_key is None:
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
        if not imagesize:
            imagesize = "256x256"
        try:
            response = openai.Image.create(
                    prompt=prompt,
                    n=num_images,
                    size=imagesize)
            image_url = response['data'][0]['url']
            return label_picturize_it, self.image_utils.url_to_image(image_url), self.image_utils.parse_image_name(response['data'])
        except openai.error.OpenAIError as error_except:
            print(f"ImageOperations create_image_from_prompt")
            print(error_except.http_status)
            print(error_except.error)
            return label_picturize_it, self.image_utils.fallback_image_implement(), self.image_utils.fallback_image_array_implement()
