import os
from CloudinaryUtil.CloudinaryClient import *  #set_folder_name,search_images, upload_image
from MongoUtil.CelebDataClient import * #get_describe, get_celebs_response
from MongoUtil.StateDataClient import *
from OpenAIUtil.ImageOperations import * #create_image_from_prompt, create_variation_from_image
from OpenAIUtil.TranscribeOperations import *  #transcribe
from OpenAIUtil.TextOperations import *
from Utils.ImageUtils import * #fallback_image_implement
from Utils.DiffusionImageGenerator import * #generate_image
from Utils.StabilityAPI import * #text_to_image
# UI Component handlers
class AskMeUIHandlers:

    def __init__(self):
        self.NO_API_KEY_ERROR="Review Configuration tab for keys/settings"
        self.LABEL_GPT_CELEB_SCREEN = "Name, Describe, Preview and Upload"
        self.fallback_image = "https://plchldr.co/i/336x280"
        #self.connection_string, self.database = self.get_private_mongo_config()
        #self.stability_api_key = self.get_stabilityai_config()
        #self.api_key, self.org_id = self.get_openai_config()
        #self.cloudinary_cloud_name, self.cloudinary_api_key, self.cloudinary_api_secret = self.get_cloudinary_config()
        
    def get_cloudinary_config(self):
        return os.getenv("CLOUDINARY_CLOUD_NAME"), os.getenv("CLOUDINARY_API_KEY"), os.getenv("CLOUDINARY_API_SECRET")

    def get_stabilityai_config(self):
        return os.getenv("STABILITY_API_KEY")

    def get_openai_config(self):
        return os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_ORG_ID")

        
    def get_private_mongo_config(self):
        return os.getenv("P_MONGODB_URI"), os.getenv("P_MONGODB_DATABASE")
        

    def set_cloudinary_config(self, cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret):        
        self.cloudinary_cloud_name = cloudinary_cloud_name
        self.cloudinary_api_key = cloudinary_api_key
        self.cloudinary_api_secret = cloudinary_api_secret

    def set_stabilityai_config(self, stability_api_key):
        self.stability_api_key = stability_api_key

    def set_openai_config(self,api_key, org_id, mongo_prompt_read_config=True):
        self.api_key = api_key
        self.org_id = org_id
        self.mongo_prompt_read_config = mongo_prompt_read_config

    def set_mongodb_config(self, mongo_config, connection_string, database):
        if not mongo_config:
            self.connection_string, self.database = self.get_private_mongo_config()
        else:
            self.connection_string = connection_string
            self.database = database
        

    def get_celebs_response_handler(self, keyword):
        image_utils = ImageUtils()
        celeb_client = CelebDataClient(self.connection_string, self.database)
        name, prompt, response, image_url, generated_image_url = celeb_client.get_celebs_response(keyword)
        try:
            if image_url and generated_image_url and response and prompt:
                return name, prompt, response, image_utils.url_to_image(image_url), image_utils.url_to_image(generated_image_url)
            elif response and prompt and (image_url and not generated_image_url):
                return name, prompt, response, image_utils.url_to_image(image_url), None
            elif response and prompt and (not image_url and generated_image_url):
                return name, prompt, response, None, image_utils.url_to_image(generated_image_url)
            elif not response and not prompt and not image_url and generated_image_url:
                return keyword, "", "", None, None
        except Exception as err:
            return f"Error {err} in AskMeUIHandlers -> Celebs Response Handler", "", "", None, None
    
    
            
    def cloudinary_search(self, folder_name):
        if not self.cloudinary_cloud_name:
            return
        if not self.cloudinary_api_key:
            return
        if not self.cloudinary_api_secret:
            return
        if not folder_name:
            return
    
        cloudinary_client = CloudinaryClient(self.cloudinary_cloud_name, self.cloudinary_api_key, self.cloudinary_api_secret)
        cloudinary_client.set_folder_name(folder_name)
        return cloudinary_client.search_images()
    
            
    def cloudinary_upload(self, folder_name, input_celeb_picture, celebrity_name):
        image_utils = ImageUtils()
        if not self.cloudinary_cloud_name:
            return "", image_utils.fallback_image_implement()
        if not self.cloudinary_api_key:
            return "", image_utils.fallback_image_implement()
        if not self.cloudinary_api_secret:
            return "", image_utils.fallback_image_implement()
        if not folder_name:
            return "", image_utils.fallback_image_implement()
        cloudinary_client = CloudinaryClient(self.cloudinary_cloud_name, self.cloudinary_api_key, self.cloudinary_api_secret)
        cloudinary_client.set_folder_name(folder_name)
        url = cloudinary_client.upload_image(input_celeb_picture, celebrity_name)    
    
        return "Uploaded - Done", image_utils.url_to_image(url)
    
    
    def generate_image_stability_ai_handler(self, name, prompt):
        if self.stability_api_key:
            stability_api = StabilityAPI(self.stability_api_key)
            prompt = f"A wallpaper photo of {name} by WLOP"
            output_generated_image=stability_api.text_to_image(name,prompt)
            return "Image generated using stability AI ", output_generated_image
        else:
            image_utils = ImageUtils()
            return self.NO_API_KEY_ERROR, image_utils.fallback_image_implement()
    
    def generate_image_diffusion_handler(self, name, prompt):
        if name:
            image_utils = ImageUtils()
            try: 
                prompt = f"A wallpaper photo of {name} by WLOP"
                image_generator = DiffusionImageGenerator()
                output_generated_image = image_generator.generate_image(name,prompt)
                return "Image generated using stabilityai/stable-diffusion-2 model", output_generated_image
            except Exception as err:
                return f"Error : {err}", image_utils.fallback_image_implement()
        else:
            return "No Name given", image_utils.fallback_image_implement()
            
    def transcribe_handler(self, audio_file):
        if not self.api_key or not audio_file:
            return self.NO_API_KEY_ERROR, self.NO_API_KEY_ERROR
        transcribe_operations = TranscribeOperations(self.api_key, self.org_id)
        return transcribe_operations.transcribe(audio_file)
    
    def create_variation_from_image_handler(self, input_image_variation, input_imagesize, input_num_images):
        if not self.api_key:
            image_utils = ImageUtils()
            return self.NO_API_KEY_ERROR, image_utils.fallback_image_implement(),image_utils.fallback_image_array_implement()
        image_operations = ImageOperations(self.api_key, self.org_id)
        return image_operations.create_variation_from_image(input_image_variation, input_imagesize, input_num_images)
    
    def create_image_from_prompt_handler(self, input_prompt, input_imagesize, input_num_images):
        if not self.api_key:
            image_utils = ImageUtils()
            return self.NO_API_KEY_ERROR, image_utils.fallback_image_implement(),image_utils.fallback_image_array_implement()
        image_operations = ImageOperations(self.api_key, self.org_id)
        return image_operations.create_image_from_prompt(input_prompt, input_imagesize, input_num_images)
    
    
    def ask_chatgpt(self, prompt, keyword, prompttype):
        if not prompt or not keyword:
            return "Prompt or keyword is required!",""
        try:        
            state_data_client = StateDataClient(self.connection_string, self.database)
            if self.mongo_prompt_read_config:
                database_prompt, database_response = state_data_client.read_description_from_prompt(keyword)
                if database_response:
                    return "Response from Database", database_response
            if self.api_key:
                operations = TextOperations(self.api_key, self.org_id)
                response_message, response = operations.chat_completion(prompt)
                state_data_client.save_prompt_response(prompt, keyword, response, prompttype)
                return response_message, response
            else:
                return self.NO_API_KEY_ERROR, ""
        except Exception as err:
            return f"Error {err} in AskMeUIHandlers -> ask_chatgpt",""


    def ask_chatgpt_summarize(self, prompt):
        if not prompt:
            return "Prompt is required!",""
        try:        
            if self.api_key:
                operations = TextOperations(self.api_key, self.org_id)
                response_message, response = operations.summarize(prompt)
                return response_message, response
            else:
                return self.NO_API_KEY_ERROR, ""
        except Exception as err:
            return f"Error {err} in AskMeUIHandlers -> ask_chatgpt_summarize",""
    
    
    
    def describe_handler(self, name, prompt, folder_name, input_celeb_real_picture, input_celeb_generated_picture):
        image_utils = ImageUtils()                  
        name = name.strip()
        if not self.api_key or not prompt or not name :            
            return f"Name or prompt is not entered or {self.NO_API_KEY_ERROR}", "", "", "", None, None
        try:
            celeb_client = CelebDataClient(self.connection_string, self.database)
            local_name, local_prompt, response, real_picture_url, generated_image_url = celeb_client.get_celebs_response(name)
                
            if self.api_key is not None and response is None and not response:
                operations = TextOperations(self.api_key, self.org_id)
                response_message, response = operations.chat_completion(prompt)   
            
            cloudinary_client = CloudinaryClient(self.cloudinary_cloud_name, self.cloudinary_api_key, self.cloudinary_api_secret)
            if real_picture_url is not None:
                cloudinary_client.set_folder_name(folder_name)
                real_picture_url = cloudinary_client.upload_image(input_celeb_real_picture, name)
            if generated_image_url is not None:
                cloudinary_client.set_folder_name("Generated")
                generated_image_url = cloudinary_client.upload_image(input_celeb_generated_picture, name)

            if real_picture_url is None:
                real_picture_url = ""
            if generated_image_url is None:
                generated_image_url = ""

            if response is not None:
                celeb_client.update_describe(name, prompt, response, real_picture_url, generated_image_url)
            
            
            return f"{self.LABEL_GPT_CELEB_SCREEN} - uploaded and saved", name, prompt, response, image_utils.url_to_image(real_picture_url), image_utils.url_to_image(generated_image_url)
        except Exception as err:
            return f"Error {err} in AskMeUIHandlers -> describe_handler", "", "", "", None, None
    
