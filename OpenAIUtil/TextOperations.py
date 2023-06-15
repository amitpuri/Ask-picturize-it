from OpenAIUtil.Operations import *

import openai

class TextOperations(Operations):
    def __init__(self, api_key: str, model_name: str, org_id: str):
        if org_id is not None:
            openai.organization = org_id
        
        if model_name is not None:
            self.model_name = model_name
        else:

            self.model_name = "gpt-3.5-turbo"
        openai.api_key = api_key 
        self.models = ["gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613", "gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613"]

    def summarize(self, prompt: str):
        try:
            if prompt is not None and openai.api_key is not None:
                completion = openai.ChatCompletion.create(
                    model=f"{self.model_name}",
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
                response = openai.Moderation.create(
                                input=f"{prompt}"
                            )
                print("Moderation response")
                print(response)
                if response["results"][0]["flagged"]=="true":                    
                    return response["results"][0]["categories"], ""
                
                if self.model_name in self.models:
                    completion = openai.ChatCompletion.create(
                        model=f"{self.model_name}",
                        messages=[
                            {"role": "user", "content": f"{prompt}"}
                        ])
                    response = completion["choices"][0]["message"].content     
                else: 
                    completion = openai.Completion.create(
                      model=f"{self.model_name}",
                      prompt=f"{prompt}"
                    )
                    response = completion["choices"][0]["text"]
                return "Response from ChatGPT", response
                

        except openai.error.OpenAIError as error_except:
            print("TextOperations TextCompletion")
            print(error_except.http_status)
            print(error_except.error)
            return error_except.error["message"], ""
