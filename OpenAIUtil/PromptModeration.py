from OpenAIUtil.Operations import *

import openai

class PromptModeration(Operations):
    def __init__(self, api_key: str, org_id: str):
        self.api_key = api_key
        if org_id is not None:
            self.org_id = org_id
   
         
    def moderation(self, prompt: str):
        try:
            if prompt is not None and self.api_key is not None:
                if self.org_id is not None:
                    openai.organization = self.org_id
                openai.api_key = self.api_key
                openai.api_version = "2020-11-07" 
                openai.api_type = "openai"
                response = openai.Moderation.create(input=f"{prompt}")
                return response["results"][0]["flagged"], response["results"][0]["categories"]
        except Exception as error_except:            
            print(f"Moderation create exception openai.error, Error {error_except} {openai.api_base} ")
            print(error_except.http_status)
            print(error_except.error)
            return False, error_except.error["message"]            