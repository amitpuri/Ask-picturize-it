from OpenAIUtil.Operations import *

import openai

class TextOperations(Operations):
    def __init__(self, api_key: str, org_id: str):
        if org_id is not None:
            openai.organization = org_id
        self.gpt_35_turbo_model = "gpt-3.5-turbo"
        openai.api_key = api_key        

    def summarize(self, prompt: str):
        try:
            if prompt is not None and openai.api_key is not None:
                completion = openai.ChatCompletion.create(
                    model=f"{self.gpt_35_turbo_model}",
                    messages=[
                        {"role": "user", "content": f"Summarize the following text :{prompt}"}
                    ])
                response =  completion["choices"][0]["message"].content     
                return "Response from ChatGPT", response
        except openai.error.OpenAIError as error_except:
            print("TextOperations summarize")
            print(error_except.http_status)
            print(error_except.error)
            return error_except.error["message"], ""
                
    def chat_completion(self, prompt: str):
        try:
            if prompt is not None and openai.api_key is not None:
                completion = openai.ChatCompletion.create(
                    model=f"{self.gpt_35_turbo_model}",
                    messages=[
                        {"role": "user", "content": f"{prompt}"}
                    ])
                response = completion["choices"][0]["message"].content     
                return "Response from ChatGPT", response
        except openai.error.OpenAIError as error_except:
            print("TextOperations TextCompletion")
            print(error_except.http_status)
            print(error_except.error)
            return error_except.error["message"], ""
