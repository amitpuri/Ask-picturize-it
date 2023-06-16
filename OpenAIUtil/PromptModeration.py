from OpenAIUtil.Operations import *

import openai

class PromptModeration(Operations):
    def __init__(self, api_key: str, org_id: str):
        if org_id is not None:
            openai.organization = org_id
   
        openai.api_key = api_key 

    def moderation(self, prompt: str):
        try:
            if prompt is not None and openai.api_key is not None:
                response = openai.Moderation.create(input=f"{prompt}")
                return response["results"][0]["flagged"], response["results"][0]["categories"]
        except openai.error.OpenAIError as error_except:
            print("Moderation create")
            print(error_except.http_status)
            print(error_except.error)
            return error_except.error["message"], ""
    
