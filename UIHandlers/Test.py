from OpenAIUtil.TextOperations import *
from OpenAIUtil.PromptModeration import *
from PalmUtil.PaLMTextOperations import *
from Utils.AskPicturizeIt import *


class Test:
    def __init__(self):
        self.ask_picturize_it = AskPicturizeIt()

    def test_handler(self, api_key, org_id, model_name, optionSelection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, google_generative_api_key, prompt):        
        if optionSelection not in ["OpenAI API","Azure OpenAI API","Google PaLM API"]:
            raise ValueError("Invalid choice!")
            
        match optionSelection:
            case  "OpenAI API":
                if api_key is None or len(api_key)==0:
                    return self.ask_picturize_it.NO_API_KEY_ERROR, ""
                else:
                    promptmoderation = PromptModeration(api_key, org_id)
                    flagged, results_categories = promptmoderation.moderation(prompt)
                    if flagged:
                        return results_categories, ""
                    else:
                        operations = TextOperations()
                        operations.set_openai_api_key(api_key)
                        operations.set_model_name(model_name)
                        if org_id:
                            operations.set_org_id(org_id) 
                        message, response = operations.chat_completion(prompt)
                        return message, response
            case "Azure OpenAI API":
                if azure_openai_key is None or len(azure_openai_key)==0:
                    return self.ask_picturize_it.NO_API_KEY_ERROR, ""
                else:
                    operations = TextOperations()        
                    operations.set_azure_openai_api_key(azure_openai_key, azure_openai_api_base, azure_openai_deployment_name)
                    message, response = operations.chat_completion(prompt)
                    return message, response
            case "Google PaLM API":
                if google_generative_api_key is None or len(google_generative_api_key)==0:
                    return self.ask_picturize_it.NO_GOOGLE_PALM_AI_API_KEY_ERROR, ""    
                else:
                    operations = PaLMTextOperations(google_generative_api_key)        
                    response = operations.generate_text(prompt)
                    return "Response from Google PaLM API", response
