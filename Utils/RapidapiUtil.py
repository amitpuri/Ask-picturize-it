'''
Rapid API extract and summarize
'''

import requests
import json

class RapidapiUtil: 
    
    def article_rapidapi_api(self, api_action, rapidapi_api_key, article_link, response_element, length=1):
        querystring = {"url": article_link}
        if api_action == "summarize":
            querystring = {"url": article_link,"length":length, 'html': "TRUE"}
            
        url = f"https://article-extractor-and-summarizer.p.rapidapi.com/{api_action}"    
        
    
        headers = {
        	"X-RapidAPI-Key": rapidapi_api_key,
        	"X-RapidAPI-Host": "article-extractor-and-summarizer.p.rapidapi.com"
        }
    
        response = requests.get(url, headers=headers, params=querystring)
        try:            
            return response.json()[response_element]
        except KeyError:
            return response["error"]  