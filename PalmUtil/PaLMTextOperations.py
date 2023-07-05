import os
import google.generativeai as palm

class PaLMTextOperations():
    def __init__(self, api_key):
        self.api_key = api_key
        
    def generate_text(self, prompt: str, 
                      model = "models/text-bison-001",
                      temperature = 0.7, 
                      candidate_count = 1,
                      top_k = 40,
                      top_p = 0.95,
                      max_output_tokens = 1024):
         
        
        palm.configure(api_key=self.api_key)
        defaults = {
                  'model': model,
                  'temperature': temperature,
                  'candidate_count': candidate_count,
                  'top_k': top_k,
                  'top_p': top_p,
                  'max_output_tokens': max_output_tokens,
                  'stop_sequences': [],
                  'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":1},{"category":"HARM_CATEGORY_TOXICITY","threshold":1},{"category":"HARM_CATEGORY_VIOLENCE","threshold":2},{"category":"HARM_CATEGORY_SEXUAL","threshold":2},{"category":"HARM_CATEGORY_MEDICAL","threshold":2},{"category":"HARM_CATEGORY_DANGEROUS","threshold":2}],
                }
        
        response = palm.generate_text(
          **defaults,
          prompt=prompt
        )
        return response.result