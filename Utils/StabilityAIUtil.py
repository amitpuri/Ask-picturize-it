from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from PIL import Image

class StabilityAIUtil:
    def __init__(self, api_key):
        available_engines = [ "stable-diffusion-xl-beta-v2-2-2",
                             "stable-diffusion-v1", 
                             "stable-diffusion-v1-5", 
                             "stable-diffusion-512-v2-0", 
                             "stable-diffusion-768-v2-0", 
                             "stable-inpainting-v1-0", 
                             "stable-inpainting-512-v2-0" ]                            
        self.engine_id = available_engines[0]
        self.api_key = api_key

        if self.api_key is None:
            raise Exception("StabilityAI API key is required.")

    def generate(self, prompt, init_image = None, num_images = 1, steps = 50):
        stability_api = client.StabilityInference(
            key=self.api_key,
            verbose=True, 
            engine=self.engine_id, 
            )
        if not init_image:
            answers = stability_api.generate(            
                prompt=prompt,
                steps=steps, 
                cfg_scale=8.0, 
                width=512, 
                height=512, 
                samples=num_images, 
                sampler=generation.SAMPLER_K_DPMPP_2M # ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m
                )
        else:
            answers = stability_api.generate(            
                prompt=prompt,
                init_image = init_image
                steps=steps, 
                cfg_scale=8.0, 
                width=512, 
                height=512, 
                samples=num_images, 
                sampler=generation.SAMPLER_K_DPMPP_2M # ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m
                )
        results = []
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    raise Exception("The prompt could not be processed the API's safety filters, modify and retry!")
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                    results.append(img)
        
        return results