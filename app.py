import json
import os

import wikipedia
import requests

import gpt3_tokenizer
import gradio as gr

import openai
import google.auth
from google.oauth2 import service_account

from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models import ChatVertexAI
from langchain.retrievers import WikipediaRetriever

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.schema import HumanMessage, SystemMessage
from ExamplesUtil.PromptGenerator import *
from MongoUtil.StateDataClient import *
from MongoUtil.CelebDataClient import *
from UIHandlers.AskMeUI import AskMeUI
from UIHandlers.Test import Test
from UIHandlers.KnowledgeBase import KnowledgeBase
from AssemblyAIUtil.AssemblyAITranscriber import AssemblyAITranscriber
from OpenAIUtil.TranscribeOperations import TranscribeOperations
from ElevenlabsUtil.ElevenlabsVoiceGenerator import ElevenlabsVoiceGenerator
from Utils.PromptOptimizer import PromptOptimizer
from Utils.StabilityAPI import StabilityAPI
from Utils.AskPicturizeIt import *
from Utils.LinkedInImageGenerator import LinkedInImageGenerator
from Utils.OpenJourneyImageGenerator import OpenJourneyImageGenerator
from Utils.RunwaymlImageGenerator import RunwaymlImageGenerator
from Utils.CompVisImageGenerator import CompVisImageGenerator
from Utils.TranscribeSpeechbrain import TranscribeSpeechbrain

#from dotenv import load_dotenv
#load_dotenv()

prompt_optimizer = PromptOptimizer()
prompt_generator =  PromptGenerator()
test = Test()
kb = KnowledgeBase()

def diffusion_models_handler(model_selection : str, prompt :str, stability_api_key: str, 
                             openai_api_key: str, openai_org_id: str, 
                             optionSelection: str, azure_openai_key: str, 
                             azure_openai_api_base: str, azure_openai_deployment_name: str, 
                             input_imagesize: str, input_num_images: int):
                                
    if model_selection not in AskPicturizeIt.diffusion_models:
        raise ValueError("Invalid choice!")
    try: 
        match model_selection:
            case  "prompthero/linkedin-diffusion":
                image_generator = LinkedInImageGenerator()
                image_result = image_generator.generate_image(prompt)
                return f"Image generated from {model_selection}", image_result
            case  "prompthero/openjourney":
                image_generator = OpenJourneyImageGenerator()
                image_result = image_generator.generate_image(prompt)
                return f"Image generated from {model_selection}", image_result
            case "runwayml/stable-diffusion-v1-5":
                image_generator = RunwaymlImageGenerator()
                image_result = image_generator.generate_image(prompt)
                return f"Image generated from {model_selection}", image_result
            case  "CompVis/stable-diffusion-v1-4":
                image_generator = CompVisImageGenerator()
                image_result = image_generator.generate_image(prompt)
                return f"Image generated from {model_selection}", image_result
            case "stability.ai":
                stability_api = StabilityAPI(stability_api_key)
                output_generated_image = stability_api.text_to_image(text_prompts = [prompt]) 
                return "Image generated using stability AI ", output_generated_image
            case "dall-e":
                uihandlers = get_AskMeUI(openai_api_key, openai_org_id, optionSelection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name)
                return uihandlers.create_image_from_prompt_handler(prompt, input_imagesize, input_num_images)
    except Exception as exception:
        print(f"Exception Name: {type(exception).__name__}")
        print(exception)
        return f" {model_selection} image_generator Error - {exception}", None

def tokenizer_calc(prompt :str):
    if prompt:
        return f"Tokenizer (tokens/characters) {gpt3_tokenizer.count_tokens(prompt)}, {len(prompt)}"

'''
Record voice, transcribe, picturize, create variations, and upload
'''

def transcribe_handler(api_key :str, org_id :str, audio_file :str):
    if audio_file: 
        uihandlers = AskMeUI()
        uihandlers.set_openai_config(api_key)
        if org_id:
            set_org_id(org_id)
        return uihandlers.transcribe_handler(audio_file)

def transcribe_whisper_large_v2(audio_file :str):
    if audio_file: 
        transcribeOperations = TranscribeOperations()
        return transcribeOperations.transcribe_whisper_large_v2(audio_file)

def try_transcribe(model_selection, audio_file :str, language :str ="en", assemblyai_api_key :str = None, openai_api_key :str = None, org_id :str = None):
    if model_selection not in AskPicturizeIt.audio_models:
        raise ValueError("Invalid choice!")
    if audio_file: 
        if language in ["en","hi","fr","es"]:
            match language:
                case  "hi": # Hindi
                    match model_selection:
                        case "speechbrain/speechbrain":
                            transcribeOperations = TranscribeSpeechbrain()
                            text, message = transcribeOperations.transcribe(audio_file)
                            return message, text
                        case "openai/whisper-1":
                            uihandlers = AskMeUI()
                            uihandlers.set_openai_config(openai_api_key)
                            if org_id:
                                uihandlers.set_org_id(org_id)
                            text, message  = uihandlers.transcribe_handler(audio_file, language)    
                            return message, text
                        case other:
                            return f"{model_selection} and {language} combination is not supported!", "" 
                case "fr": # French
                    match model_selection:
                        case "openai/whisper-1":
                            uihandlers = AskMeUI()
                            uihandlers.set_openai_config(openai_api_key)
                            if org_id:
                                uihandlers.set_org_id(org_id)
                            text, message  = uihandlers.transcribe_handler(audio_file, language)    
                            return message, text
                        case other:
                            return f"{model_selection} and {language} combination is not supported!", "" 
                case "es": # Spanish
                    match model_selection:
                        case "openai/whisper-1":
                            uihandlers = AskMeUI()
                            uihandlers.set_openai_config(openai_api_key)
                            if org_id:
                                uihandlers.set_org_id(org_id)
                            text, message  = uihandlers.transcribe_handler(audio_file, language)    
                            return message, text
                        case other:
                            return f"{model_selection} and {language} combination is not supported!", "" 
                case "en":
                    match model_selection:
                        case "assemblyai/assemblyai":
                            if not assemblyai_api_key:
                                return AskPicturizeIt.NO_ASSEMBLYAI_API_KEY_ERROR, ""                       
                            transcriber = AssemblyAITranscriber(assemblyai_api_key)
                            text = transcriber.transcribe(audio_file)
                            return text, text 
                        case "openai/whisper-1":
                            uihandlers = AskMeUI()
                            uihandlers.set_openai_config(openai_api_key)
                            if org_id:
                                uihandlers.set_org_id(org_id)
                            text, message  = uihandlers.transcribe_handler(audio_file, language)    
                            return message, text
                        case other:
                            return f"{model_selection} and {language} combination is not supported!", "" 
                case other:
                        return f"{model_selection} and {language} combination is not supported!", "" 
        else:
            return "", "No supported voice language!"

    

def get_AskMeUI(api_key :str, org_id :str, optionSelection :str, azure_openai_key :str, azure_openai_api_base :str, azure_openai_deployment_name :str):
        if optionSelection not in AskPicturizeIt.llm_api_options:
            raise ValueError("Invalid choice!")
        
        uihandlers = AskMeUI()        
        match optionSelection:
            case  "OpenAI API":
                uihandlers.set_openai_config(api_key)        
                if org_id:
                    uihandlers.set_org_id(org_id)
                return uihandlers
            case "Azure OpenAI API":
                uihandlers.set_azure_openai_config(azure_openai_key, azure_openai_api_base, azure_openai_deployment_name)
                return uihandlers
            case "Google PaLM API":
                return uihandlers

def get_private_mongo_config():
        return os.getenv("P_MONGODB_URI"), os.getenv("P_MONGODB_DATABASE")

def get_key_traits(name):
    connection_string, database = get_private_mongo_config()
    celeb_data_client = CelebDataClient(connection_string, database)
    return celeb_data_client.get_key_traits(name)

'''
Image generation 
'''

def create_image_from_prompt_handler(api_key :str, org_id :str, optionSelection :str, azure_openai_key :str, azure_openai_api_base :str, azure_openai_deployment_name :str, input_prompt :str, input_imagesize :str, input_num_images :int):    
    if api_key or azure_openai_key:
        uihandlers = get_AskMeUI(api_key, org_id, optionSelection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name)
        return uihandlers.create_image_from_prompt_handler(input_prompt, input_imagesize, input_num_images)
    else:
        return AskPicturizeIt.NO_API_KEY_ERROR, None, None

'''
Image variations 
'''

def create_variation_from_image_handler(api_key, org_id, optionSelection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, input_image_variation, input_imagesize, input_num_images):
    if api_key or azure_openai_key:
        uihandlers = get_AskMeUI(api_key, org_id, optionSelection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name)   
        return uihandlers.create_variation_from_image_handler(input_image_variation, input_imagesize, input_num_images)
    else:
        return AskPicturizeIt.NO_API_KEY_ERROR, None, None


'''
Know your Celebrity
'''


def describe_handler(api_key, org_id, model_name, optionSelection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, cloudinary_folder, mongo_config, mongo_connection_string, mongo_database, celebs_name_label, question_prompt, know_your_celeb_description, input_celeb_real_picture, input_celeb_generated_picture):
    uihandlers = get_AskMeUI(api_key, org_id, optionSelection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name)
    uihandlers.set_model_name(model_name)
    uihandlers.set_mongodb_config(mongo_config, mongo_connection_string, mongo_database)
    uihandlers.set_cloudinary_config(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret)
    return uihandlers.describe_handler(celebs_name_label, question_prompt, cloudinary_folder, know_your_celeb_description, input_celeb_real_picture, input_celeb_generated_picture)

def celeb_upload_save_real_generated_image_handler(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, cloudinary_folder, mongo_config, mongo_connection_string, mongo_database, celebs_name_label, question_prompt, know_your_celeb_description, celeb_real_photo, celeb_generated_image):
    uihandlers = AskMeUI()
    uihandlers.set_mongodb_config(mongo_config, mongo_connection_string, mongo_database)
    uihandlers.set_cloudinary_config(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret)
    return uihandlers.celeb_upload_save_real_generated_image(celebs_name_label, question_prompt, know_your_celeb_description, cloudinary_folder, celeb_real_photo, celeb_generated_image)

    
def get_celebs_response_change_handler(mongo_config, mongo_connection_string, mongo_database, celebrity, image_prompt_text, key_traits):
    return get_celebs_response(mongo_config, mongo_connection_string, mongo_database, celebrity, image_prompt_text, key_traits)
    

def get_celebs_response(mongo_config, mongo_connection_string, mongo_database, celebrity, image_prompt_text, key_traits):
    uihandlers = AskMeUI()
    uihandlers.set_mongodb_config(mongo_config, mongo_connection_string, mongo_database)
    response = None
    key_traits = get_key_traits(celebrity)
    try:
        name, prompt, response, wiki_image, generated_image_url = uihandlers.get_celebs_response_handler(celebrity)
    except:
        name = celebrity
        prompt = celebrity
        generated_image_url = None
        wiki_image = None  

    retriever = WikipediaRetriever(load_all_available_meta=True)
    docs = retriever.get_relevant_documents(query=celebrity)    
    wiki_summary = docs[0].metadata["summary"]
    #wiki_image = docs[0].metadata["image_urls"][0]    
    return name, prompt, wiki_summary, response, wiki_image, generated_image_url, f"{name}", key_traits

def celeb_summarize_handler(api_key, org_id, prompt):
    uihandlers = AskMeUI()
    uihandlers.set_openai_config(api_key)
    if org_id:
        uihandlers.set_org_id(org_id)
    return uihandlers.ask_chatgpt_summarize(prompt)

def celeb_save_description_handler(mongo_config, mongo_connection_string, mongo_database, name, prompt, description):
    uihandlers = AskMeUI()
    if name and know_your_celeb_description:     
        uihandlers.set_mongodb_config(mongo_config, mongo_connection_string, mongo_database)
        uihandlers.update_description(name, prompt, description)
        return f"ChatGPT description saved for {name}", description


def celebs_name_search_handler(api_key, org_id, model_name, optionSelection, 
                               azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, 
                               google_generative_api_key, google_project_id, google_model_name, search_text, celebs_chat_history,
                               input_language, output_language,
                              location="us-east1"):
    if optionSelection not in AskPicturizeIt.llm_api_options:
        raise ValueError("Invalid choice!")
        
    if not api_key or len(api_key.strip())==0:        
        return search_text, celebs_chat_history, AskPicturizeIt.NO_API_KEY_ERROR
    elif len(search_text.strip())<=0:
        return None, celebs_chat_history, "Error: No Input"

    celebs_chat_history = celebs_chat_history + [(search_text, None)] 
    try:
        chat = None
        match optionSelection:
            case  "OpenAI API":                    
                chat = ChatOpenAI(                                                
                    openai_api_key=api_key,
                    model=model_name,
                    temperature=0.7)
            case "Azure OpenAI API":
                chat = AzureChatOpenAI(
                    openai_api_type="azure",
                    openai_api_key=azure_openai_key,
                    openai_api_base=azure_openai_api_base,
                    deployment_name=azure_openai_deployment_name,
                    model=model_name,
                    temperature=0.7,
                    openai_api_version="2023-05-15")
            case "Google PaLM API":                    
                #TO DO AUTH
                #service_account_info = os.getenv["GOOGLE_APPLICATION_CREDENTIALS"]
                # or,
                #with open('service_service.json') as source:
                #    service_account_info = json.load(source)
                
                #credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes="googleapis.com")
                #project_id and location
                # load_credentials_from_dict with service account info
                '''
                chat = ChatVertexAI(
                    project_id=google_project_id,
                    model_name=google_model_name,
                    max_output_tokens=256,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40,
                    verbose=True)
                '''
                return None, celebs_chat_history, f"Error: The LLM provider {optionSelection} is not yet supported."  
    except Exception as exception:
        errorMessage = f"Error: For {optionSelection} - Exception Name: {type(exception).__name__} - {exception}"
        print(errorMessage)
        return None, celebs_chat_history, errorMessage
        
    try:
        if input_language in output_language:
            template = (
                    "You are a helpful assistant that answers this question."
            )
        else:
            template = (
                    "You are a helpful assistant that answers this question in {input_language} and translate to [output_language]."
            )
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        if input_language in output_language:
            llm_response = chat(
                chat_prompt.format_prompt(
                    text=search_text
                ).to_messages()
            )
        else:
            llm_response = chat(
                chat_prompt.format_prompt(
                    input_language=input_language, 
                    output_language=output_language,
                    text=search_text
                ).to_messages()
            )
        return llm_response.content, celebs_chat_history, "In progress"
    except Exception as exception:
        errorMessage = f"Error: Formating prompt and LLMing for {optionSelection} - Exception Name: {type(exception).__name__} - {exception}"
        print(errorMessage) 
        return None, celebs_chat_history, errorMessage


def celebs_name_search_history_handler(search_text, celebs_chat_history, celebs_name_search_label): 
    default_celeb_name = "John Doe"
    try:      
        if "Error" in celebs_name_search_label["label"]:
            return None, default_celeb_name, celebs_chat_history, celebs_name_search_label["label"]
        
        if search_text is not None:
            celebrity_name=search_text.replace(".", "").strip()
            if len(celebrity_name)>0:
                celebs_chat_history[-1][1] = celebrity_name
                return None, celebrity_name, celebs_chat_history, f"Review Celebrity tab for {celebrity_name} details, else rewrite your question to get a right answer."
            else:
                return None, default_celeb_name, celebs_chat_history, "Error 1: Input error!"
        else:   
            return None, default_celeb_name, celebs_chat_history, "Error 2: Input error!"
    except Exception as exception:
        errorMessage = f"celebs_name_search_history_handler -> Exception Name: {type(exception).__name__} - {exception} "
        print(errorMessage)
        return None, default_celeb_name, celebs_chat_history, errorMessage

'''
Codex
'''

def ask_chatgpt_handler(api_key, org_id, model_name, optionSelection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, mongo_config, mongo_connection_string, mongo_database, prompt, keyword):
    uihandlers = get_AskMeUI(api_key, org_id, optionSelection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name)
    uihandlers.set_model_name(model_name)
    uihandlers.set_mongodb_config(mongo_config, mongo_connection_string, mongo_database)    
    return uihandlers.ask_chatgpt(prompt, keyword,"codex")

'''
Awesome ChatGPT Prompts
'''

def awesome_prompts_handler(api_key, org_id, model_name, optionSelection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, mongo_config, mongo_connection_string, mongo_database, prompt, keyword):    
    uihandlers = get_AskMeUI(api_key, org_id, optionSelection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name)
    uihandlers.set_model_name(model_name)
    uihandlers.set_mongodb_config(mongo_config, mongo_connection_string, mongo_database)
    return uihandlers.ask_chatgpt(prompt, keyword,"awesome-prompts")


'''
Product Definition
'''

def ask_product_def_handler(api_key, org_id, model_name, optionSelection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, mongo_config, mongo_connection_string, mongo_database, prompt, keyword):
    uihandlers = get_AskMeUI(api_key, org_id, optionSelection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name)
    uihandlers.set_model_name(model_name)
    return uihandlers.ask_chatgpt(prompt, keyword,"product")


def update_final_prompt(product_fact_sheet, product_def_question, product_task_explanation):
    final_prompt = ""
    if product_fact_sheet:
        final_prompt = f"{product_task_explanation}. {product_def_question}\n\n\nTechnical specifications: \n\n\n{product_fact_sheet}"
    else:
        final_prompt = f"{product_task_explanation}. {product_def_question}"
    final_prompt = final_prompt.replace('\n\n','\n')
    return final_prompt


'''
Output and Upload
'''

def cloudinary_search(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, folder_name):
    uihandlers = AskMeUI()
    uihandlers.set_cloudinary_config(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret)
    return uihandlers.cloudinary_search(folder_name)

    
def cloudinary_upload(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, folder_name, input_celeb_picture, celebrity_name):
    uihandlers = AskMeUI()
    uihandlers.set_cloudinary_config(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret)
    return uihandlers.cloudinary_upload(folder_name, input_celeb_picture, celebrity_name)


'''
Image generation
'''

def text2image_stability_ai_handler(api_key, prompt):
    try:    
        stability_api = StabilityAPI(api_key)
        output_generated_image = stability_api.text_to_image(text_prompts = [prompt]) 
        return "Image generated using stability AI ", output_generated_image
    except Exception as err:
        return f"{err}", None
        
def image2image_stability_ai_handler(api_key, init_image, prompt = None):
    try:    
        stability_api = StabilityAPI(api_key)
        output_generated_image = stability_api.image_to_image(                        
                init_image = init_image,
                text_prompts = prompt, 
            )
        return "Image variation generated using stability AI ", output_generated_image
    except Exception as err:
        return f"{err}", None
        
def stability_ai_handler(api_key, prompt = None, init_image = None):
    if api_key:
        if init_image:
            return image2image_stability_ai_handler(api_key, init_image, prompt)
        else:
            return text2image_stability_ai_handler(api_key, prompt)
    else:
        return AskPicturizeIt.NO_STABILITYAI_API_KEY_ERROR , None

def generate_image_diffusion_handler(generate_image_prompt_text):
    uihandlers = AskMeUI()
    if generate_image_prompt_text and len(generate_image_prompt_text)>0:
        return uihandlers.generate_image_diffusion_handler("ai-generated-image", generate_image_prompt_text)
    else:
        return AskPicturizeIt.ENTER_A_PROMPT_IMAGE, None

                
# Examples fn

def get_celeb_examples(category):
    connection_string, database = get_private_mongo_config()
    celeb_data_client = CelebDataClient(connection_string, database)
    celeb_list = celeb_data_client.celeb_list(category)
    return celeb_list

def get_saved_prompts(keyword): 
    try:
        connection_string, database = get_private_mongo_config()
        state_data_client = StateDataClient(connection_string, database)
        prompt, response = state_data_client.read_description_from_prompt(keyword)        
    except:
        prompt = ""
        response = ""
        pass
    finally:
        return prompt, response

def get_keyword_prompts(prompttype):
    connection_string, database = get_private_mongo_config()
    state_data_client = StateDataClient(connection_string, database)
    saved_prompts = state_data_client.list_saved_prompts(prompttype)
    return saved_prompts

def get_input_examples():
    return prompt_generator.get_input_examples()

def get_images_examples():
    images = []
    for imagefile in prompt_generator.get_images_examples():
        images.append(imagefile["file_path"])
    return images

keyword_examples = AskPicturizeIt.KEYWORD_EXAMPLES
audio_examples = prompt_generator.get_audio_examples()
hindi_audio_examples = prompt_generator.get_audio_examples(lang = "hindi")
images_examples = get_images_examples()
input_examples = prompt_generator.get_input_examples()
product_def_keyword_examples =  get_keyword_prompts("product") 
recent_awesome_chatgpt_prompts = get_keyword_prompts("awesome-prompts")
saved_prompts = get_keyword_prompts("codex") 
saved_products =  prompt_generator.get_all_awesome_chatgpt_prompts("product")
awesome_chatgpt_prompts = prompt_generator.get_all_awesome_chatgpt_prompts()
IndianFilm_celeb_list = get_celeb_examples("Indian Film")
Hollywood_celeb_list = get_celeb_examples("Hollywood")
Business_celeb_list = get_celeb_examples("Business")
IndianFilm_celeb_examples = [celeb[0] for celeb in IndianFilm_celeb_list]
hollywood_celeb_examples = [celeb[0] for celeb in Hollywood_celeb_list]
business_celeb_examples = [celeb[0] for celeb in Business_celeb_list]
task_explanation_examples = AskPicturizeIt.TASK_EXPLANATION_EXAMPLES
product_def_question_examples = AskPicturizeIt.PRODUCT_DEF_QUESTION_EXAMPLES
article_links_examples = AskPicturizeIt.ARTICLE_LINKS_EXAMPLES
pdf_examples = kb.PDF_Examples()
youtube_links_examples = kb.YouTube_Examples()
celeb_search_questions = AskPicturizeIt.CELEB_SEARCH_QUESTIONS_EXAMPLES
                               

'''
UI Components
'''

def generated_images_gallery_on_select(evt: gr.SelectData, generated_images_gallery):    
    if evt.index >= 0:
        name = generated_images_gallery[evt.index]["name"]
        output_generated_image =  name
        return output_generated_image
    else:        
        return None

'''
test handlers
'''

def elevenlabs_test_handler(api_key: str, test_string: str, test_voice: str):
    try:
        voice_generator = ElevenlabsVoiceGenerator(api_key)
        test_audio_file = voice_generator.generate_voice(test_voice, test_string)
        return "Audio Generated", test_audio_file
    except Exception as exception:
        print(f"Exception Name: {type(exception).__name__}")
        print(exception)        
        return f"{exception}", None

def assemblyai_test_handler(api_key, test_uri):
    if not api_key:
        return AskPicturizeIt.NO_ASSEMBLYAI_API_KEY_ERROR, ""
    if not test_uri: 
        return "No audio file/uri", ""
    try:
        transcriber = AssemblyAITranscriber(api_key)
        text = transcriber.transcribe(test_uri)
        return text, text
    except Exception as exception:
        print(f"Exception Name: {type(exception).__name__}")
        print(exception)        
        return f"{exception}", None

def test_stability_ai_handler(api_key, test_style_preset, test_prompt, test_init_image, test_steps):
    if api_key:
        try:
            stability_api = StabilityAPI(api_key)
            '''
            Common parameters
                cfg_scale=7 (default) 0 - 35, How strictly the diffusion process adheres to the prompt text (higher values keep your image closer to your prompt)
                clip_guidance_preset="FAST_BLUE" - FAST_BLUE FAST_GREEN NONE SIMPLE SLOW SLOWER SLOWEST
                samples = 1, between 1-10 
                steps = 30 between 10-150
                seed between 0 .. 4294967295, default: 0 Random noise seed (omit this option or use 0 for a random seed)
            '''
            if test_init_image:    
                '''
                init_image, text_prompts, style_preset = "photographic", 
                init_image_mode IMAGE_STRENGTH or STEP_SCHEDULE
                    # IMAGE_STRENGTH 
                        # image_strength=0.35 (default) - between 0-1 or 
                    # STEP_SCHEDULE - start and end between 0-1
                        # step_schedule_start 0.65 (default) 
                        # and step_schedule_end 
                '''
                output_generated_image = stability_api.image_to_image(
                    init_image = test_init_image,
                    text_prompts = test_prompt, 
                    style_preset = test_style_preset,
                    samples = 1, 
                    steps = test_steps)
                return "Image variation generated using stability AI ", output_generated_image

            else:
                '''
                text_prompts, style_preset = "photographic", 
                height=512, width=512, 
                '''
                output_generated_image = stability_api.text_to_image(
                    text_prompts = [test_prompt], 
                    style_preset = test_style_preset,
                    samples = 1, 
                    steps = test_steps)
                return "Image generated using stability AI ", output_generated_image
        except Exception as err:
            return f"{err}", None
    else:
        return AskPicturizeIt.NO_STABILITYAI_API_KEY_ERROR, None

def clear_celeb_details():
    return "", "", "", "", None, None, None, None

with gr.Blocks(css='https://cdn.amitpuri.com/ask-picturize-it.css') as AskMeTabbedScreen:
    gr.Markdown(AskPicturizeIt.TITLE)
    with gr.Tab("Information"):
        gr.HTML(AskPicturizeIt.DESCRIPTION)
        gr.HTML(AskPicturizeIt.RESEARCH_SECTION)
        gr.HTML(AskPicturizeIt.SECTION_FOOTER)
    with gr.Tab("Configuration"):
        with gr.Tab("AI provider settings"):                       
            with gr.Group():
                with gr.Row():
                    llm_input_language = gr.Dropdown(["English"], 
                                                       value="English", label="Input Language", info="Select a language") 
                    llm_output_language = gr.Dropdown(["English"], 
                                               value="English", label="Output Language", info="Select a language") 
            with gr.Tab("OpenAI API"):
                gr.HTML(AskPicturizeIt.OPENAI_HTML)
                with gr.Row():
                    with gr.Column():                    
                        input_key = gr.Textbox(
                            label="OpenAI API Key", value=os.getenv("OPENAI_API_KEY"), type="password")
                        org_id = gr.Textbox(
                            label="OpenAI ORG ID (only for org account)", value=os.getenv("OPENAI_ORG_ID"),  type="password")  
                        openai_model = gr.Dropdown(AskPicturizeIt.openai_models, value="gpt-4", label="Model", info="Select one, for Natural language")
            with gr.Tab("Azure OpenAI API"):
                gr.HTML(AskPicturizeIt.AZURE_OPENAI_HTML)
                with gr.Row():
                    with gr.Column():                    
                        azure_openai_key = gr.Textbox(
                            label="Azure OpenAI API Key", value=os.getenv("AZURE_OPENAI_API_KEY"), type="password")
                        azure_openai_api_base = gr.Textbox(
                            label="Azure OpenAI API Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT"), type="password")
                        azure_openai_deployment_name = gr.Textbox(
                            label="Azure OpenAI API Deployment Name", value=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), type="password")
            with gr.Tab("Google API"):            
                google_model_name = gr.Dropdown(AskPicturizeIt.google_palm_models, 
                                                   value="models/text-bison-001", label="Model", info="Select one, for Natural language") 
                with gr.Tab("Google PaLM API"):
                    gr.HTML(AskPicturizeIt.GOOGLE_PALMAPI_HTML)                    
                    with gr.Row():
                        with gr.Column():
                            google_generative_api_key = gr.Textbox(
                                    label="Google Generative AI API Key", value=os.getenv("LANGUAGE_MODEL_API_KEY"), type="password")
                with gr.Tab("PaLM API on Vertex AI"):
                    gr.HTML("<p>an enterprise version of PaLM</p>")                     
                    with gr.Row():
                        with gr.Column():
                            google_project_id = gr.Textbox(
                                    label="Google Cloud Project ID", value=os.getenv("GCP_PROJECT_ID"), type="password") 
                            google_credentials = gr.Textbox(
                                    label="Google Application Credentials", value=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"), type="password")
                            google_model_locations = gr.Dropdown([
                                    "us-central1",
                                    "us-east1",
                                    "us-west1",
                                    "us-west2",
                                    "europe-west1",
                                    "asia-east1",
                                    "asia-southeast1",
                                    "australia-southeast1",
                                ], value="us-east1", label="Location", info="Select a location")
        with gr.Tab("AssemblyAI API"):
            gr.HTML(AskPicturizeIt.ASSEMBLY_AI_HTML)
            with gr.Row():
                with gr.Column():
                        assemblyai_api_key = gr.Textbox(label="AssemblyAI API Key", value=os.getenv("ASSEMBLYAI_API_KEY"), type="password")
        with gr.Tab("StabilityAI API"):
            gr.HTML(AskPicturizeIt.STABILITY_AI_HTML)
            with gr.Row():
                stabilityai_api_key = gr.Textbox(label="StabilityAI API Key", value=os.getenv("STABILITYAI_API_KEY"), type="password")
        with gr.Tab("Elevenlabs API"):
            gr.HTML(AskPicturizeIt.ELEVENLABS_HTML)
            elevenlabs_api_key = gr.Textbox(label="Elevenlabs API Key", value=os.getenv("ELEVEN_API_KEY"), type="password")
        with gr.Tab("Rapid API"):
            gr.HTML(AskPicturizeIt.RAPIDAPI_HTML)
            with gr.Row():
                with gr.Column():
                   gr.HTML(AskPicturizeIt.RAPIDAPI_ARTICLE_HTML)
                   rapidapi_api_key = gr.Textbox(label="API Key", value=os.getenv("RAPIDAPI_KEY"), type="password")  
        with gr.Tab("MongoDB"):
            gr.HTML(AskPicturizeIt.MONGODB_HTML)
            with gr.Row():
                with gr.Column(scale=3):
                    mongo_config = gr.Checkbox(label="MongoDB config", info="Use your own MongoDB", value=os.getenv("USE_MONGODB_CONFIG"))
                    mongo_connection_string = gr.Textbox(
                        label="MongoDB Connection string", value=os.getenv("MONGODB_URI"), type="password")
                with gr.Column():
                    mongo_database = gr.Textbox(
                        label="MongoDB database", value=os.getenv("MONGODB_DATABASE"))
        with gr.Tab("Cloudinary"):
            gr.HTML(AskPicturizeIt.CLOUDINARY_HTML)
            with gr.Row():
                with gr.Column():
                    cloudinary_cloud_name = gr.Textbox(
                        label="Cloudinary Cloud name", value=os.getenv("CLOUDINARY_CLOUD_NAME"))
                    cloudinary_folder = gr.Textbox(
                        label="Cloudinary folder", value=os.getenv("CLOUDINARY_FOLDER"))
                with gr.Column():
                    cloudinary_api_key = gr.Textbox(
                        label="Cloudinary API Key", value=os.getenv("CLOUDINARY_API_KEY"), type="password")
                    cloudinary_api_secret = gr.Textbox(
                        label="Cloudinary API Secret", value=os.getenv("CLOUDINARY_API_SECRET"), type="password")
        with gr.Group():
            with gr.Row():
                input_num_images = gr.Slider(minimum=1,maximum=10,step=1,
                    label="Number of Images to generate", value=1, info="OpenAI API supports 1-10 images")
                input_imagesize = gr.Dropdown(["1024x1024", "512x512", "256x256"], 
                                              value="256x256", label="Image size",
                                              info=AskPicturizeIt.imagesize_text )
    with gr.Tab("LLMs scenarios"):        
        with gr.Tab("Text-to-Text (Text Completion)"):
            openai_selection = gr.Radio(AskPicturizeIt.llm_api_options, label="Select one", info="Which service do you want to use?", value="OpenAI API")
            with gr.Row():
                with gr.Column():                    
                    test_string = gr.Textbox(
                        label="Try String", value=AskPicturizeIt.TEST_MESSAGE, lines=2)
                    test_string_response = gr.Textbox(
                        label="Response")
                    test_string_output_info = gr.Label(value="Output Info", label="Info")
                    test_button = gr.Button("Try it")
        with gr.Tab("Audio-to-Text"):
            gr.HTML(AskPicturizeIt.ASSEMBLY_AI_HTML)
            audio_model_selection = gr.Radio(AskPicturizeIt.audio_models, label="Select one", info="Which model do you want to use?", value="openai/whisper-1")
            with gr.Row():
                with gr.Column(scale=2):                    
                    speechbrain_test_upload = gr.Audio(label="Record or Upload", type="filepath", source="upload")
                    assemblyai_test_uri = gr.Audio(label="Audio to Text", type="filepath", source="microphone")                            
                    gr.Examples(
                        examples=hindi_audio_examples,                   
                        label="Select one from Hindi Audio Examples and Transcribe",
                        examples_per_page=5,
                        inputs=assemblyai_test_uri)                        
                    gr.Examples(
                        examples=audio_examples,                   
                        label="Select one from English Audio Examples and Transcribe",
                        examples_per_page=5,
                        inputs=assemblyai_test_uri)                    
                with gr.Column(scale=1):
                    assemblyai_speechbrain_test_string = gr.Textbox(label="Transcription", lines=5)
                    with gr.Accordion("Options..", open=True):
                        audio_lang_selection = gr.Dropdown(["en","hi"], label="Select one", info="Audio Language", value="en")
                    with gr.Row():                            
                        assemblyai_test_button = gr.Button("Try transcribe")
                        assemblyai_speechbrain_clear = gr.Button("Clear")
                        assemblyai_test_string_output_info = gr.Label(value="Output Info", label="Info")                
        with gr.Tab("Text-to-Audio"):        
            text2audio_selection = gr.Radio(AskPicturizeIt.text2audio_medium, label="Select one", info="Which medium do you want to use?", value="elevanlabs")
            with gr.Tab("Elevenlabs"):                   
                gr.HTML(AskPicturizeIt.ELEVENLABS_HTML)
                elevenlabs_voice = gr.Dropdown(AskPicturizeIt.elevenlabs_voices, value="Bella", label="Voice", info="Select a voice to generate audio")
            with gr.Row():
                with gr.Column():
                   elevenlabs_test_string = gr.Textbox(label="Text to Audio string", value=AskPicturizeIt.ELEVENLABS_TEST_MESSAGE, lines=2)
                   elevenlabs_test_string_output_info = gr.Label(value="Output Info", label="Info")
                   elevenlabs_test_button = gr.Button("Try Generating audio")
                   elevenlabs_test_audio_file = gr.Audio(label="Play the generated audio",type="filepath", value ="audio/english/AI as a tool that can augment and empower us, rather than compete or replace us.mp3")
        with gr.Tab("Text-to-Image"):    
            gr.HTML(AskPicturizeIt.DIFFUSION_MODELS_HTML)
            with gr.Row():
                with gr.Column(scale=1):
                    diffusion_model_selection = gr.Radio(AskPicturizeIt.diffusion_models, label="Select one", info="Which model do you want to use?", value="prompthero/linkedin-diffusion")
                    diffusion_test_string = gr.Textbox(label="Prompt", value="a lnkdn photography of Sam Altman")
                    diffusion_test_button = gr.Button("Try it")
                    diffusion_output_info = gr.Label(value="Output Info", label="Info")
                with gr.Column(scale=3):
                    diffusion_output_photo = gr.Image(label="Generated Image",  type="filepath")                    
            gr.Examples(
                examples=AskPicturizeIt.coolest_midjourney_prompts,                   
                label="Select one and try it",
                examples_per_page=10,
                inputs=diffusion_test_string)
        with gr.Tab("Image-to-Image"):
            text2image_selection = gr.Radio(AskPicturizeIt.text2image_medium, label="Select one", info="Which medium do you want to use?", value="StabilityAI")
            with gr.Tab("Stability AI"):                   
                gr.HTML(AskPicturizeIt.STABILITY_AI_HTML)
                with gr.Row():
                    stabilityai_style_preset = gr.Dropdown(AskPicturizeIt.style_presets, 
                                           value="digital-art", label="Style preset", info="Select one style preset")            
                    stabilityai_steps = gr.Slider(minimum=10, maximum=150, step=10, label="Number of diffusion steps to run", value=30, info="Diffusion steps")
            image2image_string = gr.Textbox(label="Prompt", value="panda mad scientist mixing sparkling chemicals digital art")            
            with gr.Column(scale=2):
                image2image_photo = gr.Image(label="Input Image",  type="filepath", value="images/generated-image-panda.png")                    
            with gr.Column(scale=1):
                gr.Examples(
                    examples=images_examples,
                    label="Select one from Image Examples and get variation",
                    inputs=[image2image_photo],
                    examples_per_page=10,
                    outputs=image2image_photo,
                )
            image2image_output_photo = gr.Image(label="output Image",  type="filepath")
            with gr.Row():
                with gr.Column():
                    image2image_button = gr.Button("Try it")
                    image2image_output_info = gr.Label(value="Output Info", label="Info")
        with gr.Tab("Text-to-Video"):
            gr.HTML(AskPicturizeIt.TEXT_TO_VIDEO_HTML)
            gr.HTML("Work in progress....")           
    with gr.Tab("Record, transcribe, picturize and upload"):
        gr.HTML("<p>Record voice, transcribe a prompt, picturize the prompt, create variations, and upload in Output tab</p>")
        with gr.Tab("Whisper(whisper-1)"):
            with gr.Row():
                with gr.Column(scale=3):                    
                    audio_file = gr.Audio(
                        label="Upload Audio, or Record to describe what you want to picturize and click on Transcribe",
                        source="microphone",
                        value = "audio/english/AI as a tool that can augment and empower us, rather than compete or replace us.mp3",
                        type="filepath"
                    )
                with gr.Column(scale=2):
                    gr.Examples(
                        examples=audio_examples,                   
                        label="Select one from Audio Examples and Transcribe",
                        examples_per_page=6,
                        inputs=audio_file)
                    transcribe_button = gr.Button("Transcribe via Whisper")  
                    transcribe_whisper_large_v2_button = gr.Button("Transcribe via openai/whisper-large-v2")                     
            input_transcriptionprompt = gr.Label(label="Transcription Text")
        with gr.Tab("Image generation"):
            input_prompt = gr.Textbox(label="Prompt Text to describe what you want to picturize?", lines=7)
            with gr.Row():                
                with gr.Column(scale=1):                    
                    optimize_prompt_chatgpt_button = gr.Button("Optimize Prompt")
                    generate_button = gr.Button("Picture it via DALL-E")
                    generate_image_diffusion_button = gr.Button("*via stable-diffusion-2 model")
                    generate_image_stability_ai_button = gr.Button("via Stability AI")
                    label_generate_image_diffusion = gr.Label(value="* takes 30-50 mins on CPU", label="Warning") 
                with gr.Column(scale=5):
                    gr.Examples(
                        examples=input_examples,
                        label="Select one from Prompt Examples",
                        inputs=[input_prompt],
                        examples_per_page=10,
                        outputs=input_prompt,
                    )
                    label_picturize_it = gr.Label(value="Prompt in your words and picturize it", label="Info")
        with gr.Tab("Image variation"):
            with gr.Row():
                input_image_variation = gr.Image(
                    label="Input Image", type="filepath")
                gr.Examples(
                    examples=images_examples,
                    label="Select one from Image Examples and get variation",
                    inputs=[input_image_variation],
                    examples_per_page=10,
                    outputs=input_image_variation,
                )
            with gr.Row():
                label_get_variation = gr.Label(
                        value="Get variation of your favorite celebs", label="Info")                
                with gr.Column():
                    generate_variations_button = gr.Button("Generate a variation via DALL-E")
                    generate_variations_image_stability_ai_button = gr.Button("via Stability AI")
        with gr.Tab("Output"):
            with gr.Row():            
                with gr.Column(scale=4):
                    with gr.Accordion("Generated Gallery", open=False):
                        generated_images_gallery = gr.Gallery(
                                        label="Generated Images", preview="False", columns=4)
                    output_generated_image = gr.Image(label="Preview Image",  type="filepath")
                with gr.Column(scale=1):
                    output_cloudinary_button = gr.Button("Get images from Cloudinary")
                    generate_more_variations_button = gr.Button("More variations via DALL-E")
                    generate_more_image_stability_ai_button = gr.Button("via Stability AI")
                    name_variation_it = gr.Textbox(label="Name variation to upload")   
                    variation_cloudinary_upload = gr.Button("Upload to Cloudinary")
            label_upload_variation = gr.Label(value="Upload output", label="Output Info")
    with gr.Tab("Use cases"):
        usecases_llm_selection = gr.Radio(AskPicturizeIt.llm_api_options, label="Select one", info="Which service do you want to use?", value="OpenAI API")
        with gr.Tab("Know your Celebrity"):
            with gr.Tab("GPT Search"):
                with gr.Row():
                    with gr.Column(scale=7):                        
                        celebs_name_chatbot = gr.Chatbot()
                        celebs_name_search = gr.Textbox(label="Question")                        
                    with gr.Column(scale=1):   
                        celebs_name_search_label = gr.Label(value="GPT search output info", label="Info")
                        gr.Examples(
                            label="Search Questions",
                            examples=celeb_search_questions,
                            examples_per_page=6,
                            inputs=[celebs_name_search],
                            outputs=[celebs_name_search],
                        )
                        celebs_name_search_clear = gr.Button("Clear")
            with gr.Tab("Celebrity"):
                with gr.Row():
                    with gr.Column(scale=4):
                        celebs_name_label = gr.Textbox(label="Celebrity") 
                        question_prompt = gr.Textbox(label="Prompt", lines=2)
                        key_traits = gr.Textbox(label="Key traits", lines=5)
                        with gr.Accordion("Celebrity Examples, select one from here", open=True):
                            with gr.Tab("Indian Film"):
                                with gr.Row():
                                    gr.Examples(
                                        label="Select one from a celebrity",
                                        examples=IndianFilm_celeb_examples,
                                        examples_per_page=100,
                                        inputs=[celebs_name_label],
                                        outputs=[question_prompt, key_traits],                                             
                                    )
                            with gr.Tab("Hollywood"):
                                with gr.Row():
                                    gr.Examples(
                                        label="Select one from a celebrity",
                                        examples=hollywood_celeb_examples,
                                        examples_per_page=100,
                                        inputs=[celebs_name_label],
                                        outputs=[question_prompt, key_traits],                                          
                                    )
                            with gr.Tab("Business"):
                                with gr.Row():
                                    gr.Examples(
                                        label="Select one from a celebrity",
                                        examples=business_celeb_examples,
                                        examples_per_page=100,
                                        inputs=[celebs_name_label],
                                        outputs=[question_prompt, key_traits],                
                                    )                        
                    with gr.Column(scale=1):
                        clear_celeb_details_button = gr.Button("Clear")                
                        generate_image_prompt_text = gr.Textbox(label="Image generation prompt")
                        label_describe_gpt = gr.Label(value="Generate or Upload Image to Save", label="Info")
                        with gr.Accordion("Options..", open=True):
                            generate_celeb_image_stability_ai_button = gr.Button("via Stability AI")
                            celeb_variation_button = gr.Button("variation from the real photo (DALL-E 2)")                                
                with gr.Row():
                    celeb_real_photo = gr.Image(label="Real Photo",  type="filepath")                        
                    celeb_generated_image = gr.Image(label="AI Generated Image",  type="filepath")
                with gr.Row():
                    with gr.Column(scale=1):
                        know_your_celeb_description_wiki = gr.Textbox(label="Wiki summary", lines=13)
                    with gr.Column(scale=1):
                        know_your_celeb_description = gr.Textbox(label="Description from OpenAI ChatGPT", lines=7)
                        with gr.Row():                    
                            celeb_summarize_copy_button = gr.Button("Summarize Wiki output, copy to Description")
                            celeb_save_description_button = gr.Button("Save Description")
                            describe_button = gr.Button("Describe via ChatGPT and Save")
                            celeb_upload_save_real_generated_image_button = gr.Button("Upload, Save real & generated image")                    
                label_upload_here = gr.Label(value=AskPicturizeIt.LABEL_GPT_CELEB_SCREEN, label="Info")            
        
        with gr.Tab("Summarizer"):
            with gr.Tab("KB Search"):
                with gr.Row():
                    with gr.Column(scale=4):                    
                        keyword_search = gr.Textbox(label="Keyword", placeholder="Search Arxiv, YouTube, wikipedia?")
                        gr.Examples(
                                label="Keyword examples",
                                examples=keyword_examples,
                                examples_per_page=150,
                                inputs=[keyword_search],
                                outputs=[keyword_search],
                        )
                    with gr.Column(scale=1):  
                        max_results = gr.Slider(minimum=10,maximum=100,step=5,label="Max Results", value=10, info="Search results output")
                        select_medium = gr.Dropdown(["YouTube", "Arxiv","Wikipedia"], label="Search in", value="Arxiv", type="index" )
                        keyword_search_button = gr.Button("Search")    
                keyword_search_output = gr.JSON() 
            with gr.Tab("Summarizer via LLM using LangChain"):
                gr.HTML(AskPicturizeIt.LANGCHAIN_TEXT)
                with gr.Tab("YouTube"):                    
                    with gr.Row():
                        with gr.Column(scale=4):                    
                            youtube_link = gr.Textbox(label="Enter YouTube link")
                            youtube_title = gr.Textbox(label="Title")
                            gr.Examples(
                                    label="YouTube examples",
                                    examples=youtube_links_examples,
                                    fn=kb.youtube_search_data_by_uri,
                                    run_on_click=True,
                                    cache_examples = False,
                                    examples_per_page=25,
                                    inputs=[youtube_link],
                                    outputs=[youtube_link,youtube_title],
                            )
                        with gr.Column(scale=1):  
                            youtube_transcribe_summarize_info_label = gr.Label(value="Transcribe and summarize Output info", label="Info")
                            youtube_transcribe_button = gr.Button("Transcribe")
                            youtube_summarize_button = gr.Button("Summarize")
                    youtube_transcribe_summary = gr.Textbox(label="YouTube summary response", lines=10)
                with gr.Tab("PDF"):
                    with gr.Row():
                        with gr.Column(scale=4):                    
                            pdf_link = gr.Textbox(label="Enter PDF link")
                            pdf_title = gr.Textbox(label="Title")
                            pdf_summary = gr.Textbox(label="Summary")
                            pdf_pages = gr.Textbox(label="Pages")
                            gr.Examples(
                                    label="PDF examples",
                                    fn=kb.pdf_search_data_by_uri,
                                    run_on_click=True,
                                    cache_examples = False,
                                    examples=pdf_examples,
                                    examples_per_page=25,
                                    inputs=[pdf_link],
                                    outputs=[pdf_link,pdf_title,pdf_summary, pdf_pages],
                            )
                        with gr.Column(scale=1):  
                            pdf_summarize_info_label = gr.Label(value="PDF summarize Output info", label="Info")
                            pdf_read_contents_button= gr.Button("Read PDF")
                            pdf_summarize_button = gr.Button("Summarize PDF")
                    pdf_summary = gr.Textbox(label="PDF response", lines=10) 
            with gr.Tab("Article Extractor and Summarizer"):
                gr.HTML(AskPicturizeIt.RAPIDAPI_ARTICLE_HTML)
                with gr.Row():                
                    with gr.Column(scale=4):                    
                        article_link = gr.Textbox(label="Enter Article link")
                        gr.Examples(
                                label="Article examples",
                                examples=article_links_examples,
                                examples_per_page=25,
                                inputs=[article_link],
                                outputs=[article_link],
                        )
                    with gr.Column(scale=1):  
                        article_summarize_extract_info_label = gr.Label(value="Article Extractor and Summarizer Output info", label="Info")
                        article_summarize_length = gr.Slider(minimum=1, maximum=20, step=1, label="Length", value=1, info="Length")
                        article_article_summarize_button = gr.Button("Summarize")
                        article_article_extract_button = gr.Button("Extract")            
                article_summary = gr.Code(label="Article response", language="html", lines=10)
    with gr.Tab("Ask GPT"):
        with gr.Row():
            with gr.Column(): 
                gpt_search = gr.Textbox(label="Keyword")
                with gr.Row():
                    keyword_search_prompt = gr.Textbox(label="Prompt")
                    keyword_search_response = gr.Textbox(label="Response")
            with gr.Column():    
                with gr.Tab("Recent Codex"):
                        with gr.Row():
                            gr.Examples(
                                    label="Recent keywords",
                                    fn=get_saved_prompts,
                                    examples=saved_prompts,
                                    examples_per_page=10,
                                    inputs=[gpt_search],
                                    outputs=[keyword_search_prompt, keyword_search_response],   
                                    cache_examples = False,
                                    run_on_click=True,
                                )  
                with gr.Tab("Recent Awesome Prompts"):
                    with gr.Row():
                        gr.Examples(
                                label="Recent Prompts",
                                fn=get_saved_prompts,
                                examples=recent_awesome_chatgpt_prompts,
                                examples_per_page=50,
                                inputs=[gpt_search],
                                outputs=[keyword_search_prompt, keyword_search_response],
                                cache_examples = False,
                                run_on_click=True,
                            )
                with gr.Tab("Recent Product definition"):
                    with gr.Row():
                        gr.Examples(
                                    label="Recent product definitions",
                                    fn=get_saved_prompts,
                                    examples=product_def_keyword_examples,
                                    examples_per_page=5,
                                    inputs=[gpt_search],
                                    outputs=[keyword_search_prompt, keyword_search_response],  
                                    cache_examples = False,
                                    run_on_click=True,
                        )
        with gr.Tab("Ask Codex"):
            with gr.Row():
                with gr.Column(scale=4):
                    ask_keyword = gr.Textbox(label="Keyword", lines=1)
                    ask_prompt = gr.Textbox(label="Prompt", lines=5)                
                with gr.Column(scale=1):   
                    label_codex_here = gr.Label(value="Ask Codex, Write a better code", label="Info")
                    ask_chatgpt_button = gr.Button("Ask ChatGPT")                
            with gr.Row():
                keyword_response_code = gr.Code(label="Code", language="python", lines=7)               
        with gr.Tab("🧠 Awesome ChatGPT Prompts"):
            gr.HTML(AskPicturizeIt.AWESOME_CHATGPT_PROMPTS)
            with gr.Row():
                with gr.Column(scale=4):
                    awesome_chatgpt_act = gr.Textbox(label="Act")
                    awesome_chatgpt_prompt = gr.Textbox(label="Awesome ChatGPT Prompt", lines=5)                        
                with gr.Column(scale=1):
                    label_awesome_chatgpt_here = gr.Label(value="See examples below", label="Info")
                    ask_awesome_chatgpt_button = gr.Button("Ask ChatGPT")                                
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Examples(
                        label="Awesome ChatGPT Prompts",
                        fn=prompt_generator.get_awesome_chatgpt_prompt,
                        examples=awesome_chatgpt_prompts,
                        examples_per_page=50,
                        inputs=[awesome_chatgpt_act],
                        outputs=[awesome_chatgpt_prompt],
                        cache_examples = False,
                        run_on_click=True,
                    )
            with gr.Row():            
                awesome_chatgpt_response = gr.Textbox(label="Response", lines=20)
        with gr.Tab("Product definition"):
            gr.HTML(AskPicturizeIt.PRODUCT_DEFINITION)
            with gr.Row():
                with gr.Column(scale=4):
                    product_def_keyword = gr.Textbox(label="Keyword")
                    product_def_final_prompt = gr.Textbox(label="Prompt", lines=10)
                with gr.Column(scale=1):                
                    with gr.Row():                            
                        product_def_info_label = gr.Label(value="See examples below", label="Info")
                        product_def_ask_button = gr.Button("Ask ChatGPT")  
            with gr.Row():                    
                with gr.Column(scale=4):
                    product_fact_sheet = gr.Textbox(label="Product Fact sheet", lines=25)
                    product_task_explanation = gr.Textbox(label="Task explanation", lines=5)
                    product_def_question = gr.Textbox(label="Question", lines=5)
                with gr.Column(scale=1):
                    gr.HTML(AskPicturizeIt.PRODUCT_DEFINITION_INFO)
                    gr.Examples(
                                label="Product Fact sheet examples",
                                fn=prompt_generator.get_awesome_chatgpt_prompt,
                                examples=saved_products,
                                examples_per_page=3,
                                inputs=[product_def_keyword],
                                outputs=[product_fact_sheet],
                                cache_examples = False,
                                run_on_click=True,
                            )
                    gr.Examples(
                        label="Task explanation examples",
                        examples=task_explanation_examples,
                        examples_per_page=10,
                        inputs=[product_task_explanation],
                        outputs=[product_task_explanation],
                    )
                    gr.Examples(
                        label="Question examples",
                        examples=product_def_question_examples,
                        examples_per_page=6,
                        inputs=[product_def_question],
                        outputs=[product_def_question],
                    )
            with gr.Row():
                product_def_response = gr.Textbox(label="Response", lines=10)
        with gr.Tab("Picturize Product"):
            with gr.Row():
                with gr.Column(scale=4):
                    product_def_image_prompt = gr.Textbox(label="Enter Image creation Prompt", lines=5)
                    product_def_generated_image = gr.Image(label="AI Generated Image",  type="filepath")
                with gr.Column(scale=1):                
                    optimize_prompt_product_def_button = gr.Button("Optimize Prompt")
                    product_def_generate_button = gr.Button("Picturize it")
                    product_def_variations_button = gr.Button("More variations")
                    product_def_image_info_label = gr.Label(value="Picturize it info", label="Info")
    with gr.Tab("DISCLAIMER"):
        gr.Markdown(AskPicturizeIt.DISCLAIMER)
    gr.HTML(AskPicturizeIt.FOOTER)

    speechbrain_test_upload.upload(
        fn = lambda speechbrain_test_upload, assemblyai_test_uri : speechbrain_test_upload,  
        inputs = [speechbrain_test_upload, assemblyai_test_uri], 
        outputs = [assemblyai_test_uri], 
        queue=False
        )

    speechbrain_test_upload.change(
        fn = lambda speechbrain_test_upload, assemblyai_test_uri : speechbrain_test_upload,  
        inputs = [speechbrain_test_upload, assemblyai_test_uri], 
        outputs = [assemblyai_test_uri], 
        queue=False
        )

    
    assemblyai_speechbrain_clear.click(
        fn = lambda speechbrain_test_upload, 
                       assemblyai_test_uri, 
                       assemblyai_speechbrain_test_string, 
                       audio_model_selection,
                       audio_lang_selection,
                       assemblyai_test_string_output_info : [None, None, None, 1, "en", "Output Info"], 
        inputs = [speechbrain_test_upload, assemblyai_test_uri, assemblyai_speechbrain_test_string, audio_model_selection, audio_lang_selection, assemblyai_test_string_output_info],
        outputs = [speechbrain_test_upload, assemblyai_test_uri, assemblyai_speechbrain_test_string, audio_model_selection, audio_lang_selection, assemblyai_test_string_output_info],
        queue=False)

    clear_celeb_details_button.click(
        clear_celeb_details,
        inputs=[],
        outputs=[celebs_name_label, question_prompt, know_your_celeb_description_wiki, know_your_celeb_description, celeb_real_photo, celeb_generated_image, generate_image_prompt_text, key_traits]
    )

    celebs_name_search_clear.click(
        fn = lambda: None, 
        inputs = None, 
        outputs = celebs_name_chatbot, 
        queue=False)
    
    
    diffusion_test_button.click(
        fn=diffusion_models_handler,
        inputs=[diffusion_model_selection, diffusion_test_string, stabilityai_api_key, input_key, org_id, openai_selection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, input_imagesize, input_num_images],
        outputs=[diffusion_output_info, diffusion_output_photo]
    )

    assemblyai_test_button.click(
        fn=try_transcribe,
        inputs=[audio_model_selection, assemblyai_test_uri, audio_lang_selection, assemblyai_api_key, input_key, org_id],
        outputs=[assemblyai_test_string_output_info, assemblyai_speechbrain_test_string]
    )
    
    elevenlabs_test_button.click(
        fn=elevenlabs_test_handler,
        inputs=[elevenlabs_api_key, elevenlabs_test_string, elevenlabs_voice],
        outputs=[elevenlabs_test_string_output_info, elevenlabs_test_audio_file]
    )

    test_button.click(
        fn=test.test_handler,
        inputs=[input_key, org_id, openai_model, openai_selection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, google_generative_api_key, test_string],
        outputs=[test_string_output_info, test_string_response]
    )

        
    image2image_button.click(
        fn=test_stability_ai_handler,
        inputs=[stabilityai_api_key, stabilityai_style_preset, image2image_string, image2image_photo, stabilityai_steps],
        outputs=[image2image_output_info, image2image_output_photo]
    )
    
    youtube_transcribe_button.click(
        fn=kb.youtube_transcribe_handler,
        inputs=[youtube_link],
        outputs=[youtube_transcribe_summarize_info_label, youtube_transcribe_summary]
    )

    keyword_search_button.click(
        fn=kb.kb_search, 
        inputs=[keyword_search,select_medium, max_results], 
        outputs=keyword_search_output
    )
    
    pdf_summarize_button.click(
        fn=kb.pdf_summarize_contents_handler,
        inputs=[usecases_llm_selection, input_key, org_id, openai_model, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, pdf_link],
        outputs=[pdf_summarize_info_label, pdf_summary]
    )

    pdf_read_contents_button.click(
        fn=kb.pdf_read_contents_handler,
        inputs=[pdf_link],
        outputs=[pdf_summarize_info_label, pdf_summary]
    )
    
    youtube_summarize_button.click(
        fn=kb.youtube_summarizer_handler,
        inputs=[usecases_llm_selection, input_key, org_id, openai_model, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, youtube_link],
        outputs=[youtube_transcribe_summarize_info_label, youtube_transcribe_summary]
    )
    
    article_article_summarize_button.click(
        fn=kb.article_summarize_handler,        
        inputs=[rapidapi_api_key, article_link, article_summarize_length], 
        outputs=[article_summary, article_summarize_extract_info_label]
    )

    article_article_extract_button.click(
        fn=kb.article_extract_handler,        
        inputs=[rapidapi_api_key, article_link], 
        outputs=[article_summary, article_summarize_extract_info_label]
    )

    optimize_prompt_product_def_button.click(
        fn=prompt_optimizer.generate_optimized_prompt,
        inputs=[product_def_image_prompt],
        outputs=[product_def_image_prompt]
    )
    
    celebs_name_search.submit(
        celebs_name_search_handler,
        inputs=[input_key, org_id, openai_model, usecases_llm_selection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, google_generative_api_key, google_project_id, google_model_name, celebs_name_search, celebs_name_chatbot, llm_input_language, llm_output_language, google_model_locations],
        outputs=[celebs_name_search, celebs_name_chatbot, celebs_name_search_label]).then(
        celebs_name_search_history_handler, 
        inputs=[celebs_name_search, celebs_name_chatbot, celebs_name_search_label], 
        outputs=[celebs_name_search, celebs_name_label, celebs_name_chatbot, celebs_name_search_label]

    )

    celeb_upload_save_real_generated_image_button.click(
        celeb_upload_save_real_generated_image_handler,
        inputs=[cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, cloudinary_folder, mongo_config, mongo_connection_string, mongo_database, celebs_name_label, question_prompt, know_your_celeb_description, celeb_real_photo, celeb_generated_image],
        outputs=[label_upload_here, celebs_name_label, question_prompt, know_your_celeb_description, celeb_real_photo, celeb_generated_image]

    )

    celeb_save_description_button.click(
        celeb_save_description_handler,
        inputs=[mongo_config, mongo_connection_string, mongo_database, celebs_name_label, question_prompt, know_your_celeb_description],
        outputs=[label_upload_here, know_your_celeb_description]
    )
    
    celeb_summarize_copy_button.click(
        celeb_summarize_handler,
        inputs=[input_key, org_id, know_your_celeb_description_wiki],
        outputs=[label_upload_here, know_your_celeb_description]

    )
    
    
    celebs_name_label.change(
        fn=get_celebs_response_change_handler,
        inputs=[mongo_config, mongo_connection_string, mongo_database, celebs_name_label, generate_image_prompt_text, key_traits],
        outputs=[celebs_name_label, question_prompt, know_your_celeb_description_wiki, know_your_celeb_description, celeb_real_photo, celeb_generated_image, generate_image_prompt_text, key_traits]
    )

    product_def_image_prompt.change(
        fn=tokenizer_calc,
        inputs=[product_def_image_prompt],
        outputs=[product_def_image_info_label]        
    )
    
    product_fact_sheet.change(
        fn=update_final_prompt,
        inputs=[product_fact_sheet, product_def_question, product_task_explanation],
        outputs=[product_def_final_prompt]        
    )

    product_def_question.change(
        fn=update_final_prompt,
        inputs=[product_fact_sheet, product_def_question, product_task_explanation],
        outputs=[product_def_final_prompt]        
    )

    product_task_explanation.change(
        fn=update_final_prompt,
        inputs=[product_fact_sheet, product_def_question, product_task_explanation],
        outputs=[product_def_final_prompt]        
    )
    
    question_prompt.change(
        fn=tokenizer_calc,
        inputs=[generate_image_prompt_text],
        outputs=[label_describe_gpt]        
    )
    
    awesome_chatgpt_prompt.change(
        fn=tokenizer_calc,
        inputs=[awesome_chatgpt_prompt],
        outputs=[label_awesome_chatgpt_here]        
    )
    
    product_def_final_prompt.change(
        fn=tokenizer_calc,
        inputs=[product_def_final_prompt],
        outputs=[product_def_info_label]
    )
    
    input_prompt.change(
        fn=tokenizer_calc,
        inputs=[input_prompt],
        outputs=[label_picturize_it]        
    )
    
    ask_prompt.change(
        fn=tokenizer_calc,
        inputs=[ask_prompt],
        outputs=[label_codex_here]        
    )
    
    generate_celeb_image_stability_ai_button.click(
        fn=stability_ai_handler,
        inputs=[stabilityai_api_key, generate_image_prompt_text, celeb_real_photo],
        outputs=[label_describe_gpt, celeb_generated_image]
    )
    
    generate_image_diffusion_button.click(
        fn=generate_image_diffusion_handler,
        inputs=[input_prompt],
        outputs=[label_picturize_it, output_generated_image]
    )

    optimize_prompt_chatgpt_button.click(
        fn=prompt_optimizer.generate_optimized_prompt,
        inputs=[input_prompt],
        outputs=[input_prompt]
    )
       

    output_cloudinary_button.click(
        fn=cloudinary_search,
        inputs=[cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, cloudinary_folder],
        outputs=[generated_images_gallery]
    )

    generated_images_gallery.select(
        fn=generated_images_gallery_on_select,
        inputs=[generated_images_gallery],
        outputs=[output_generated_image]
    )

    variation_cloudinary_upload.click(
        fn=cloudinary_upload,
        inputs=[cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, cloudinary_folder, output_generated_image, name_variation_it],
        outputs=[label_upload_variation,output_generated_image]
    )

    transcribe_button.click(
        fn=transcribe_handler,
        inputs=[input_key, org_id, audio_file],
        outputs=[input_transcriptionprompt, input_prompt]
    )

    transcribe_whisper_large_v2_button.click(
        fn=transcribe_whisper_large_v2,
        inputs=[audio_file],
        outputs=[input_transcriptionprompt, input_prompt]
    )
    
    
    generate_button.click(
        fn=create_image_from_prompt_handler,
        inputs=[input_key, org_id, openai_selection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, input_prompt, input_imagesize, input_num_images],
        outputs=[label_picturize_it, output_generated_image, generated_images_gallery]
    )
    
    
    ask_awesome_chatgpt_button.click(
        fn=awesome_prompts_handler,
        inputs=[input_key, org_id, openai_model, openai_selection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, mongo_config, mongo_connection_string, mongo_database, awesome_chatgpt_prompt, awesome_chatgpt_act],
        outputs=[label_awesome_chatgpt_here, awesome_chatgpt_response]
    )

    ask_chatgpt_button.click(
        fn=ask_chatgpt_handler,
        inputs=[input_key, org_id, openai_model, openai_selection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, mongo_config, mongo_connection_string, mongo_database, ask_prompt, ask_keyword],
        outputs=[label_codex_here, keyword_response_code]
    )    

    product_def_ask_button.click(
        fn=ask_product_def_handler,
        inputs=[input_key, org_id, openai_model, openai_selection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, mongo_config, mongo_connection_string, mongo_database, product_def_final_prompt, product_def_keyword],
        outputs=[product_def_info_label, product_def_response]
    )   


    product_def_generate_button.click(
        fn=create_image_from_prompt_handler,
        inputs=[input_key, org_id, openai_selection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, product_def_image_prompt, input_imagesize, input_num_images],
        outputs=[product_def_image_info_label, product_def_generated_image, generated_images_gallery]

    )
    
    celeb_variation_button.click(
        fn=create_variation_from_image_handler,
        inputs=[input_key, org_id, openai_selection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, celeb_real_photo, input_imagesize, input_num_images],
        outputs=[label_upload_here, celeb_generated_image, generated_images_gallery]
    )
    
    generate_variations_button.click(
        fn=create_variation_from_image_handler,
        inputs=[input_key, org_id, openai_selection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, input_image_variation, input_imagesize, input_num_images],
        outputs=[label_get_variation, output_generated_image, generated_images_gallery]
    )

    generate_more_variations_button.click(
        fn=create_variation_from_image_handler,
        inputs=[input_key, org_id, openai_selection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, output_generated_image, input_imagesize, input_num_images],
        outputs=[label_upload_variation, output_generated_image, generated_images_gallery]
    )

    generate_image_stability_ai_button.click(
        fn=text2image_stability_ai_handler,
        inputs=[stabilityai_api_key, input_prompt],
        outputs=[label_picturize_it, output_generated_image]
    )
    
    generate_variations_image_stability_ai_button.click(
        fn=stability_ai_handler,
        inputs=[stabilityai_api_key, input_image_variation],
        outputs=[label_get_variation, output_generated_image]
    )

    generate_more_image_stability_ai_button.click(
        fn=image2image_stability_ai_handler,
        inputs=[stabilityai_api_key, output_generated_image],
        outputs=[label_get_variation, output_generated_image]
    )

    product_def_variations_button.click(
        fn=create_variation_from_image_handler,
        inputs=[input_key, org_id, openai_selection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, product_def_generated_image, input_imagesize, input_num_images],
        outputs=[product_def_image_info_label, product_def_generated_image, generated_images_gallery]
    )

    describe_button.click(
        fn=describe_handler,
        inputs=[input_key, org_id, openai_model, openai_selection, azure_openai_key, azure_openai_api_base, azure_openai_deployment_name, cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, cloudinary_folder, mongo_config, mongo_connection_string, mongo_database, celebs_name_label, question_prompt, know_your_celeb_description, celeb_real_photo, celeb_generated_image],
        outputs=[label_upload_here, celebs_name_label, question_prompt, know_your_celeb_description, celeb_real_photo, celeb_generated_image]        
    )
    

if __name__ == "__main__":
    AskMeTabbedScreen.launch()
