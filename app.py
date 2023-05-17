import json
import os
import random
import time

import gpt3_tokenizer
import wikipedia
import requests
import gradio as gr
from ExamplesUtil.CelebPromptGenerator import *
from MongoUtil.StateDataClient import *
from MongoUtil.CelebDataClient import *
from UIHandlers import AskMeUIHandlers
from Utils.Optimizers import Prompt_Optimizer
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import torch
from transformers import pipeline



#from dotenv import load_dotenv

#load_dotenv()

TITLE = '# [Ask-me-to-picturize-it](https://github.com/amitpuri/Ask-me-to-picturize-it)'



DESCRIPTION = """<strong>This space uses following:</strong>
   <p>
   <ul>

   <li>OpenAI API Whisper(whisper-1) <a href='https://openai.com/research/whisper'>https://openai.com/research/whisper</a></li>
   <li>DALL-E <a href='https://openai.com/product/dall-e-2'>https://openai.com/product/dall-e-2</a></li>
   <li>GPT(gpt-3.5-turbo) <a href='https://openai.com/product/gpt-4'>https://openai.com/product/gpt-4</a></li>

   <li>Cloudinary <a href='https://cloudinary.com/documentation/python_quickstart'>https://cloudinary.com/documentation/python_quickstart</a></li>
   <li>Gradio App <a href='https://gradio.app/docs'>https://gradio.app/docs</a> in Python and MongoDB</li>
   <li>Uses Prompt optimizer <a href='https://huggingface.co/microsoft/Promptist'>https://huggingface.co/microsoft/Promptist</a></li>
   <li>Uses stabilityai/stable-diffusion-2-1 <a href='https://huggingface.co/stabilityai/stable-diffusion-2-1'>https://huggingface.co/stabilityai/stable-diffusion-2-1</a></li>
   <li>Uses Stability AI <a href='https://stability.ai'>https://stability.ai</a></li>
   <li>LangChain OpenAI <a href='https://js.langchain.com/docs/getting-started/guide-llm'>https://js.langchain.com/docs/getting-started/guide-llm</a></li>
   <li>Uses Article Extractor and Summarizer on Rapid API <a href='https://rapidapi.com'>https://rapidapi.com</a></li>   
   </ul>
   </p>
 """
RESEARCH_SECTION = """
   <p><strong>Research papers to read</strong>

   </p>
   <p>
   <ul>
   <li><p>Attention Is All You Need <a href='https://arxiv.org/abs/1706.03762'>https://arxiv.org/abs/1706.03762</a></p></li>
   </ul>
   </p>
"""
SECTION_FOOTER = """
   <p>Note: Only PNG is supported here, as of now</p>
   <p>Visit <a href='https://ai.amitpuri.com'>https://ai.amitpuri.com</a></p>
"""
DISCLAIMER = """MIT License

Copyright (c) 2023 Amit Puri

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights

to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is

furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
FOOTER = """<div class="footer">

                    <p>by <a href="https://www.amitpuri.com" style="text-decoration: underline;" target="_blank">Amit Puri</a></p>
            </div>            
        """


AWESOME_CHATGPT_PROMPTS = """
Credits 🧠 Awesome ChatGPT Prompts <a href='https://github.com/f/awesome-chatgpt-prompts'>https://github.com/f/awesome-chatgpt-prompts</a>
"""


PRODUCT_DEFINITION = "<p>Define a product by prompt, picturize it, get variations, save it with a keyword for later retrieval. Credits <a href='https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers'>https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers</a></p>"

LABEL_GPT_CELEB_SCREEN = "Select, Describe, Generate AI Image, Upload and Save"

"""
Whisper
"""
MODEL_NAME = "openai/whisper-large-v2"
device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

all_special_ids = pipe.tokenizer.all_special_ids
transcribe_token_id = all_special_ids[-5]
translate_token_id = all_special_ids[-6]

def transcribe_whisper_large_v2(audio_file, task="transcribe"):    
    pipe.model.config.forced_decoder_ids = [[2, transcribe_token_id if task=="transcribe" else translate_token_id]]
    text = pipe(audio_file)["text"]
    return text, text

'''
Reusable functions
'''

WIKI_REQUEST = 'http://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles='

def get_wikimedia_image(keyword):
    if keyword:
        try:
            result = wikipedia.search(keyword, results = 1)
        except wikipedia.exceptions.WikipediaException as exception:
            print(f"Exception Name: {type(exception).__name__}")
            print(exception)
            result = None
            pass
        wikipedia.set_lang('en')
        try:
            if result is not None:
                wkpage = wikipedia.WikipediaPage(title = result[0])
            else:
                wkpage = None
        except wikipedia.exceptions.WikipediaException as exception:
            print(f"Exception Name: {type(exception).__name__}")
            print(exception)
            wkpage = None
            pass
        if wkpage is not None:
            title = wkpage.title
            response  = requests.get(WIKI_REQUEST+title)
            json_data = json.loads(response.text)
            try:
                image_link = list(json_data['query']['pages'].values())[0]['original']['source']
                return image_link
            except:
                return None

def get_wiki_page_summary(keyword):
    if keyword:
        try:
            return wikipedia.page(keyword).summary
        except wikipedia.exceptions.PageError:
            return f"No page for this keyword {keyword}"
        except Exception as exception:
            print(f"Exception Name: {type(exception).__name__}")
            print(exception)

    
        
def generate_optimized_prompt(plain_text):
    prompt_optimizer = Prompt_Optimizer()
    return prompt_optimizer.generate_optimized_prompt(plain_text);
	
def tokenizer_calc(prompt):
    if prompt:
        return f"Tokenizer (tokens/characters) {gpt3_tokenizer.count_tokens(prompt)}, {len(prompt)}"


def get_private_mongo_config():
    return os.getenv("P_MONGODB_URI"), os.getenv("P_MONGODB_DATABASE")


def use_case_slider_handler(use_case_slider):    
    use_case_options={      
      1:'Ask Codex',
      2:'Awesome ChatGPT Prompts',
      3:'Product definition'
      }
    return use_case_options.get(use_case_slider,"Invalid use case")

    

'''
Record voice, transcribe, picturize, create variations, and upload
'''


def get_audio_examples():
    prompt_generator = CelebPromptGenerator()
    return prompt_generator.get_audio_examples()


def transcribe_handler(api_key, org_id, audio_file):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_openai_config(api_key, org_id)
    return uihandlers.transcribe_handler(audio_file)
    

'''
Image generation Examples
'''

def get_input_examples():
    prompt_generator = CelebPromptGenerator()
    return prompt_generator.get_input_examples()



def create_image_from_prompt_handler(api_key, org_id, input_prompt, input_imagesize, input_num_images):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_openai_config(api_key, org_id)
    return uihandlers.create_image_from_prompt_handler(input_prompt, input_imagesize, input_num_images)


'''
Image variations Examples
'''

def get_images_examples():
    prompt_generator = CelebPromptGenerator()
    return prompt_generator.get_images_examples()




def create_variation_from_image_handler(api_key, org_id, input_image_variation, input_imagesize, input_num_images):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_openai_config(api_key, org_id)
    return uihandlers.create_variation_from_image_handler(input_image_variation, input_imagesize, input_num_images)


'''
Know your Celebrity
'''


def describe_handler(api_key, org_id, mongo_prompt_read_config, cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, cloudinary_folder, mongo_config, mongo_connection_string, mongo_database, celebs_name_label, question_prompt, know_your_celeb_description, input_celeb_real_picture, input_celeb_generated_picture):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_openai_config(api_key, org_id, mongo_prompt_read_config)
    uihandlers.set_mongodb_config(mongo_config, mongo_connection_string, mongo_database)
    uihandlers.set_cloudinary_config(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret)
    return uihandlers.describe_handler(celebs_name_label, question_prompt, cloudinary_folder, know_your_celeb_description, input_celeb_real_picture, input_celeb_generated_picture)

def celeb_upload_save_real_generated_image_handler(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, cloudinary_folder, mongo_config, mongo_connection_string, mongo_database, celebs_name_label, question_prompt, know_your_celeb_description, celeb_real_photo, celeb_generated_image):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_mongodb_config(mongo_config, mongo_connection_string, mongo_database)
    uihandlers.set_cloudinary_config(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret)
    return uihandlers.celeb_upload_save_real_generated_image(celebs_name_label, question_prompt, know_your_celeb_description, cloudinary_folder, celeb_real_photo, celeb_generated_image)


def get_celebrity_detail_from_wiki(celebrity):
    celebrity_name = f"{celebrity}"
    return get_wiki_page_summary(celebrity_name),get_wikimedia_image(celebrity)

    
def get_celebs_response_change_handler(mongo_config, mongo_connection_string, mongo_database, celebrity, image_prompt_text, key_traits):
    return get_celebs_response(mongo_config, mongo_connection_string, mongo_database, celebrity, image_prompt_text, key_traits)

def get_internal_celeb_name(celebrity):
    for celeb in IndianFilm_celeb_list:
        if celeb[0]==celebrity:
            return celeb[1]
    for celeb in Hollywood_celeb_list:
        if celeb[0]==celebrity:
            return celeb[1]
    for celeb in Business_celeb_list:
        if celeb[0]==celebrity:
            return celeb[1]
    

def get_celebs_response(mongo_config, mongo_connection_string, mongo_database, celebrity, image_prompt_text, key_traits):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_mongodb_config(mongo_config, mongo_connection_string, mongo_database)
    internal_celeb_name = get_internal_celeb_name(celebrity)
    wiki_summary = get_wiki_page_summary(internal_celeb_name)
    key_traits = get_key_traits(celebrity)
    try:
        name, prompt, response, wiki_image, generated_image_url = uihandlers.get_celebs_response_handler(celebrity)
        if wiki_image is None:
            wiki_image = get_wikimedia_image(celebrity)
        return name, prompt, wiki_summary, response, wiki_image, generated_image_url, f"A realistic photo of {name}", key_traits
    except:
        response = None
        generated_image_url = None
        wiki_image = get_wikimedia_image(celebrity)
        return celebrity, f"Write a paragraph on {celebrity}", wiki_summary, "", wiki_image, None, f"A realistic photo of {celebrity}", key_traits
        pass
    

def clear_celeb_details():
    return "", "", "", "", None, None, None, None


def celeb_summarize_handler(api_key, org_id, prompt):
    uihandlers = AskMeUIHandlers()    
    uihandlers.set_openai_config(api_key, org_id)
    return uihandlers.ask_chatgpt_summarize(prompt)

def celeb_save_description_handler(mongo_config, mongo_connection_string, mongo_database, name, prompt, description):
    if name and know_your_celeb_description:     
        uihandlers = AskMeUIHandlers() 
        uihandlers.set_mongodb_config(mongo_config, mongo_connection_string, mongo_database)
        uihandlers.update_description(name, prompt, description)
        return f"ChatGPT description saved for {name}", description

def get_celeb_examples(category):
    connection_string, database = get_private_mongo_config()           
    celeb_data_client = CelebDataClient(connection_string, database)
    celeb_list = celeb_data_client.celeb_list(category)
    return celeb_list

def get_key_traits(name):
    connection_string, database = get_private_mongo_config()
    celeb_data_client = CelebDataClient(connection_string, database)
    return celeb_data_client.get_key_traits(name)

def celebs_name_search_handler(input_key, search_text, celebs_chat_history):
    if not input_key or len(input_key.strip())==0:        
        return None, celebs_chat_history, "Review Configuration tab for keys/settings", "OPENAI_API_KEY is missing"

    if len(search_text.strip())>0:
        os.environ["OPENAI_API_KEY"] = input_key
        celebs_chat_history = celebs_chat_history + [(search_text, None)] 
        llm = OpenAI(temperature=0.7)
        llm_response = llm(search_text)        
        return llm_response, celebs_chat_history, "In progress"
    else:
        return None, celebs_chat_history, "No Input"

def celebs_name_search_history_handler(search_text, celebs_chat_history):    
    if search_text:
        celebrity_name = search_text.replace(".", "").strip()
        celebs_chat_history[-1][1] = celebrity_name
        return None, celebrity_name, celebs_chat_history, f"Review Celebrity tab for {celebrity_name} details"
    else:
        return None, "John Doe", celebs_chat_history, "Review Configuration tab for keys/settings", "OPENAI_API_KEY is missing or No input"


'''
Codex
'''

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



def ask_chatgpt_handler(api_key, org_id, mongo_prompt_read_config, mongo_config, mongo_connection_string, mongo_database, prompt, keyword):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_mongodb_config(mongo_config, mongo_connection_string, mongo_database)
    uihandlers.set_openai_config(api_key, org_id, mongo_prompt_read_config)
    return uihandlers.ask_chatgpt(prompt, keyword,"codex")

'''
Awesome ChatGPT Prompts
'''

def get_awesome_chatgpt_prompts(awesome_chatgpt_act):
    prompt_generator = CelebPromptGenerator()
    awesome_chatgpt_prompt = prompt_generator.get_awesome_chatgpt_prompt(awesome_chatgpt_act)
    return awesome_chatgpt_act, awesome_chatgpt_prompt



def awesome_prompts_handler(api_key, org_id, mongo_prompt_read_config, mongo_config, mongo_connection_string, mongo_database, prompt, keyword):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_openai_config(api_key, org_id, mongo_prompt_read_config)
    uihandlers.set_mongodb_config(mongo_config, mongo_connection_string, mongo_database)
    return uihandlers.ask_chatgpt(prompt, keyword,"awesome-prompts")



'''
Product Definition
'''


def ask_product_def_handler(api_key, org_id, mongo_prompt_read_config, mongo_config, mongo_connection_string, mongo_database, prompt, keyword):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_mongodb_config(mongo_config, mongo_connection_string, mongo_database)
    uihandlers.set_openai_config(api_key, org_id, mongo_prompt_read_config)
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
Rapid API extract and summarize
'''

def article_rapidapi_api(api_action, rapidapi_api_key, article_link, length=1):
    querystring = {"url": article_link}
    if api_action == "summarize":
        querystring = {"url": article_link,"length":length, 'html': "TRUE"}
        
    url = f"https://article-extractor-and-summarizer.p.rapidapi.com/{api_action}"    
    

    headers = {
    	"X-RapidAPI-Key": rapidapi_api_key,
    	"X-RapidAPI-Host": "article-extractor-and-summarizer.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    
    return  response.json()

def article_summarize_handler(rapidapi_api_key, article_link, length):
    if rapidapi_api_key:
        response = article_rapidapi_api("summarize", rapidapi_api_key, article_link, length)
        return response["summary"], ""
    else:
        return "","Review Configuration tab for keys/settings", "RAPIDAPI_KEY is missing or No input"
    
def article_extract_handler(rapidapi_api_key, article_link):
    if rapidapi_api_key:
        response = article_rapidapi_api("extract", rapidapi_api_key, article_link)
        return response["content"], ""
    else:
        return "","Review Configuration tab for keys/settings", "RAPIDAPI_KEY is missing or No input"

    
# Examples fn

task_explanation_examples = ["""Your task is to help a marketing team create a description for a retail website of a product based on a technical fact sheet.
           
Write a product description based on the information provided in the technical specifications delimited by triple backticks."""]

product_def_question_examples = ["Limit answer to 50 words", 
                             "Limit answer to 100 words", 
                             "Write the answer in bullet points",
                             "Write the answer in 2/3 sentences",
                             "Write the answer in one line TLDR with the fewest words"
                            ]

article_links_examples = ["https://time.com/6266679/musk-ai-open-letter", 
                          "https://futureoflife.org/open-letter/ai-open-letter",
                          "https://github.com/openai/CLIP",
                          "https://arxiv.org/abs/2103.00020",
                          "https://spacy.io/usage/spacy-101",
                          "https://developers.google.com/machine-learning/gan/gan_structure",
                          "https://developers.google.com/machine-learning/gan/generative"]

prompt_generator = CelebPromptGenerator()
audio_examples = prompt_generator.get_audio_examples()
images_examples = prompt_generator.get_images_examples()
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
prompt_character = PromptTemplate(
    input_variables=["character_name","program_name"],
    template="What is the name of the actor acted as {character_name} in {program_name}, answer without any explanation and return only the actor's name?")

prompt_bond_girl = PromptTemplate(
    input_variables=["movie_name"],
    template="Who was Bond girl co-star in {movie_name}?, answer without any explanation and return only the actor's name?")


celeb_search_questions = [prompt_character.format(character_name="James Bond",program_name="Casino Royale"),
                        prompt_character.format(character_name="James Bond",program_name="Die Another Day"),
                        prompt_character.format(character_name="James Bond",program_name="Never Say Never Again"),
                        prompt_character.format(character_name="Patrick Jane",program_name="The Mentalist"),
                        prompt_character.format(character_name="Raymond Reddington",program_name="The Blacklist"),
                        prompt_bond_girl.format(movie_name="GoldenEye"),
                        prompt_bond_girl.format(movie_name="Die Another Day")]
                               
'''
Output and Upload
'''

def cloudinary_search(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, folder_name):
    uihandlers = AskMeUIHandlers()    
    uihandlers.set_cloudinary_config(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret)
    return uihandlers.cloudinary_search(folder_name)

    
def cloudinary_upload(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, folder_name, input_celeb_picture, celebrity_name):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_cloudinary_config(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret)
    return uihandlers.cloudinary_upload(folder_name, input_celeb_picture, celebrity_name)


'''
Image generation
'''


def generate_image_stability_ai_handler(stability_api_key, celebs_name_label, generate_image_prompt_text):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_stabilityai_config(stability_api_key)
    return uihandlers.generate_image_stability_ai_handler(celebs_name_label, generate_image_prompt_text)

    
def generate_image_diffusion_handler(celebs_name_label, generate_image_prompt_text):
    uihandlers = AskMeUIHandlers()
    return uihandlers.generate_image_diffusion_handler(celebs_name_label, generate_image_prompt_text)
    

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


with gr.Blocks(css='https://cdn.amitpuri.com/ask-picturize-it.css') as AskMeTabbedScreen:
    gr.Markdown(TITLE)
    with gr.Tab("Information"):
        gr.HTML(DESCRIPTION)
        gr.HTML(RESEARCH_SECTION)
        gr.HTML(SECTION_FOOTER)
    with gr.Tab("Configuration"):
        with gr.Tab("OpenAI API"):
            gr.HTML("Sign up for API Key here <a href='https://platform.openai.com'>https://platform.openai.com</a>")
            with gr.Row():
                with gr.Column():                    
                    input_key = gr.Textbox(
                        label="OpenAI API Key", value=os.getenv("OPENAI_API_KEY"), type="password")
                    org_id = gr.Textbox(
                        label="OpenAI ORG ID (only for org account)", value=os.getenv("OPENAI_ORG_ID"))
                    mongo_prompt_read_config = gr.Checkbox(label="Read ChatGPT response from MongoDB, if available", info="Prompt read", value="true")
        with gr.Tab("MongoDB"):
            gr.HTML("Sign up here <a href='https://www.mongodb.com/cloud/atlas/register'>https://www.mongodb.com/cloud/atlas/register</a>")            
            with gr.Row():
                with gr.Column(scale=3):
                    mongo_config = gr.Checkbox(label="MongoDB config", info="Use your own MongoDB", value=os.getenv("USE_MONGODB_CONFIG"))
                    mongo_connection_string = gr.Textbox(
                        label="MongoDB Connection string", value=os.getenv("MONGODB_URI"), type="password")
                with gr.Column():
                    mongo_database = gr.Textbox(
                        label="MongoDB database", value=os.getenv("MONGODB_DATABASE"))
        with gr.Tab("Cloudinary"):
            gr.HTML("Sign up here <a href='https://cloudinary.com'>https://cloudinary.com</a>")
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
        with gr.Tab("Stability API"):
            gr.HTML("Sign up here <a href='https://platform.stability.ai'>https://platform.stability.ai</a>")
            with gr.Row():
               with gr.Column():
                   stability_api_key = gr.Textbox(label="Stability API Key", value=os.getenv("STABILITY_API_KEY"), type="password")   
        with gr.Tab("Rapid API"):
            gr.HTML("Sign up here <a href='https://rapidapi.com'>https://rapidapi.com</a>")
            with gr.Row():
                with gr.Column():
                   rapidapi_api_key = gr.Textbox(label="Article extractor and summarizer API Key", value=os.getenv("RAPIDAPI_KEY"), type="password")   
        with gr.Row():
                input_num_images = gr.Slider(minimum=1,maximum=10,step=1,
                    label="Number of Images to generate", value=1, info="OpenAI API supports 1-10 images")
                input_imagesize = gr.Dropdown(["1024x1024", "512x512", "256x256"], value="256x256", label="Image size",
                                              info="Select one, use download for image size from Image generation/variation Output tab")
    with gr.Tab("Record, transcribe, picturize and upload"):
        gr.HTML("<p>Record voice, transcribe a prompt, picturize the prompt, create variations, and upload in Output tab</p>")
        with gr.Tab("Whisper(whisper-1)"):
            with gr.Row():
                with gr.Column(scale=3):                    
                    audio_file = gr.Audio(
                        label="Upload Audio, or Record to describe what you want to picturize and click on Transcribe",
                        source="microphone",
                        type="filepath"
                    )
                with gr.Column(scale=2):
                    gr.Examples(
                        examples=audio_examples,                   
                        label="Select one from Audio Examples and Transcribe",
                        fn=get_audio_examples,
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
                with gr.Column(scale=5):
                    gr.Examples(
                        examples=input_examples,
                        label="Select one from Prompt Examples",
                        fn=get_input_examples,
                        examples_per_page=5,
                        inputs=input_prompt
                    )
                    label_picturize_it = gr.Label(value="Prompt in your words and picturize it", label="Info")
        with gr.Tab("Image variation"):
            with gr.Row():
                input_image_variation = gr.Image(
                    label="Input Image", type="filepath")
                gr.Examples(
                    examples=images_examples,
                    label="Select one from Image Examples and get variation",
                    fn=get_images_examples,
                    examples_per_page=9,
                    inputs=input_image_variation
                )
            with gr.Row():
                label_get_variation = gr.Label(
                        value="Get variation of your favorite celebs", label="Info")                
                with gr.Column():
                    generate_variations_button = gr.Button("Generate a variation via DALL-E")
    with gr.Tab("Use cases"):
        with gr.Tab("Know your Celebrity"):
            with gr.Tab("GPT Search"):
                with gr.Row():
                    with gr.Column(scale=7):                        
                        celebs_name_chatbot = gr.Chatbot()
                        celebs_name_search = gr.Textbox(label="Question")
                    with gr.Column(scale=1):    

                        gr.Examples(
                            label="Search Questions",
                            examples=celeb_search_questions,
                            examples_per_page=6,
                            inputs=[celebs_name_search],
                            outputs=[celebs_name_search],
                        )
                        celebs_name_search_clear = gr.Button("Clear")
                        celebs_name_search_label = gr.Label(value="GPT search output info", label="Info")
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
                        generate_image_prompt_text = gr.Textbox(label="Image generation prompt", value="A realistic photo")
                        label_describe_gpt = gr.Label(value="Generate or Upload Image to Save", label="Info")
                        with gr.Accordion("Options..", open=True):
                            generate_image_stability_ai_button = gr.Button("via Stability AI")
                            generate_image_diffusion_button = gr.Button("*via stable-diffusion-2 model")
                            label_generate_image_diffusion = gr.Label(value="* takes 30-50 mins on CPU", label="Warning") 
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
                label_upload_here = gr.Label(value=LABEL_GPT_CELEB_SCREEN, label="Info")     
        with gr.Tab("Ask GPT"):
            with gr.Row():
                with gr.Column(): 
                    keyword_search = gr.Textbox(label="Keyword")
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
                                        inputs=[keyword_search],
                                        outputs=[keyword_search_prompt, keyword_search_response],   
                                        cache_examples=True,
                                    )  
                    with gr.Tab("Recent Awesome Prompts"):
                        with gr.Row():
                            gr.Examples(
                                    label="Recent Prompts",
                                    fn=get_saved_prompts,
                                    examples=recent_awesome_chatgpt_prompts,
                                    examples_per_page=50,
                                    inputs=[keyword_search],
                                    outputs=[keyword_search_prompt, keyword_search_response],   
                                    cache_examples=True,
                                )
                    with gr.Tab("Recent Product definition"):
                        with gr.Row():
                            gr.Examples(
                                        label="Recent product definitions",
                                        fn=get_saved_prompts,
                                        examples=product_def_keyword_examples,
                                        examples_per_page=5,
                                        inputs=[keyword_search],
                                        outputs=[keyword_search_prompt, keyword_search_response],   
                                        cache_examples=True,
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
                gr.HTML(AWESOME_CHATGPT_PROMPTS)
                with gr.Row():
                    with gr.Column(scale=4):
                        awesome_chatgpt_act = gr.Textbox(label="Act")
                        awesome_chatgpt_prompt = gr.Textbox(label="Awesome ChatGPT Prompt", lines=5)                        
                    with gr.Column(scale=1):
                        label_awesome_chatgpt_here = gr.Label(value="See examples below", label="Info")
                        ask_awesome_chatgpt_button = gr.Button("Ask ChatGPT")                
                with gr.Row():            
                        awesome_chatgpt_response = gr.Textbox(label="Response", lines=20)
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Examples(
                            label="Awesome ChatGPT Prompts",
                            fn=get_awesome_chatgpt_prompts,
                            examples=awesome_chatgpt_prompts,
                            examples_per_page=50,
                            inputs=[awesome_chatgpt_act],
                            outputs=[awesome_chatgpt_act, awesome_chatgpt_prompt],
                            cache_examples=True,
                        )
            with gr.Tab("Product definition"):
                gr.HTML(PRODUCT_DEFINITION)
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
                        gr.HTML("<p>Prompt builder, <br><br> Step 1 - Select a fact sheet, <br><br> Step 2 - Select a task and <br><br> Step 3 - Select a question to build it <br><br> Step 4 - Click Ask ChatGPT</p>")
                        gr.Examples(
                                    label="Product Fact sheet examples",
                                    fn=get_awesome_chatgpt_prompts,
                                    examples=saved_products,
                                    examples_per_page=3,
                                    inputs=[product_def_keyword],
                                    outputs=[product_def_keyword,product_fact_sheet],
                                    cache_examples=True,
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
        with gr.Tab("Article Extractor and Summarizer"):
            gr.HTML("Article Extractor and Summarizer API on RapidAPI <a href='https://rapidapi.com/restyler/api/article-extractor-and-summarizer'>https://rapidapi.com/restyler/api/article-extractor-and-summarizer</a>")
            with gr.Row():                
                with gr.Column(scale=4):                    
                    article_link = gr.Textbox(label="Enter Article link")
                    gr.Examples(
                            label="Article examples",
                            examples=article_links_examples,
                            examples_per_page=15,
                            inputs=[article_link],
                            outputs=[article_link],
                    )
                with gr.Column(scale=1):  
                    article_summarize_extract_info_label = gr.Label(value="Article Extractor and Summarizer Output info", label="Info")
                    article_summarize_length = gr.Slider(minimum=1, maximum=20, step=1, label="Length", value=1, info="Length")
                    article_article_summarize_button = gr.Button("Summarize")
                    article_article_extract_button = gr.Button("Extract")            
            article_summary = gr.Code(label="Article response", language="html", lines=10)
    with gr.Tab("Output"):
            with gr.Row():            
                with gr.Column(scale=4):
                    with gr.Accordion("Generated Gallery", open=False):
                        generated_images_gallery = gr.Gallery(
                                        label="Generated Images").style(preview="False", columns=[4])
                    output_generated_image = gr.Image(label="Preview Image",  type="filepath")
                with gr.Column(scale=1):
                    output_cloudinary_button = gr.Button("Get images from Cloudinary")
                    generate_more_variations_button = gr.Button("More variations via DALL-E")
                    name_variation_it = gr.Textbox(label="Name variation to upload")   
                    variation_cloudinary_upload = gr.Button("Upload to Cloudinary")
            label_upload_variation = gr.Label(value="Upload output", label="Output Info")
    with gr.Tab("DISCLAIMER"):
        gr.Markdown(DISCLAIMER)
    gr.HTML(FOOTER)


    celebs_name_search_clear.click(lambda: None, None, celebs_name_chatbot, queue=False)

    article_article_summarize_button.click(
        article_summarize_handler,        
        inputs=[rapidapi_api_key, article_link, article_summarize_length], 
        outputs=[article_summary, article_summarize_extract_info_label]
    )

    article_article_extract_button.click(
        article_extract_handler,        
        inputs=[rapidapi_api_key, article_link], 
        outputs=[article_summary, article_summarize_extract_info_label]
    )
    
    celebs_name_search.submit(
        celebs_name_search_handler,
        inputs=[input_key, celebs_name_search, celebs_name_chatbot],
        outputs=[celebs_name_search, celebs_name_chatbot, celebs_name_search_label]).then(
        celebs_name_search_history_handler, 
        inputs=[celebs_name_search, celebs_name_chatbot], 
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
    
    
    clear_celeb_details_button.click(
        clear_celeb_details,
        inputs=[],
        outputs=[celebs_name_label, question_prompt, know_your_celeb_description_wiki, know_your_celeb_description, celeb_real_photo, celeb_generated_image, generate_image_prompt_text, key_traits]
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
    
    optimize_prompt_product_def_button.click(
        generate_optimized_prompt,
        inputs=[product_def_image_prompt],
        outputs=[product_def_image_prompt]
    )

    ask_awesome_chatgpt_button.click(
        awesome_prompts_handler,
        inputs=[input_key, org_id, mongo_prompt_read_config, mongo_config, mongo_connection_string, mongo_database, awesome_chatgpt_prompt, awesome_chatgpt_act],
        outputs=[label_awesome_chatgpt_here, awesome_chatgpt_response]
    )

    ask_chatgpt_button.click(
        ask_chatgpt_handler,
        inputs=[input_key, org_id, mongo_prompt_read_config, mongo_config, mongo_connection_string, mongo_database, ask_prompt, ask_keyword],
        outputs=[label_codex_here, keyword_response_code]
    )    

    product_def_ask_button.click(
        ask_product_def_handler,
        inputs=[input_key, org_id, mongo_prompt_read_config, mongo_config, mongo_connection_string, mongo_database, product_def_final_prompt, product_def_keyword],
        outputs=[product_def_info_label, product_def_response]
    )   
    
    product_def_generate_button.click(
        create_image_from_prompt_handler,
        inputs=[input_key, org_id, product_def_image_prompt, input_imagesize, input_num_images],
        outputs=[product_def_image_info_label, product_def_generated_image, generated_images_gallery]

    )

    product_def_variations_button.click(
        create_variation_from_image_handler,
        inputs=[input_key, org_id, product_def_generated_image, input_imagesize, input_num_images],
        outputs=[product_def_image_info_label, product_def_generated_image, generated_images_gallery]
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

    product_def_generate_button.click(
        create_image_from_prompt_handler,
        inputs=[input_key, org_id, product_def_final_prompt, input_imagesize, input_num_images],
        outputs=[product_def_info_label, product_def_generated_image, generated_images_gallery]
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
    
    generate_image_stability_ai_button.click(
        generate_image_stability_ai_handler,
        inputs=[stability_api_key, celebs_name_label, generate_image_prompt_text],
        outputs=[label_upload_here,celeb_generated_image]

    )
    
    generate_image_diffusion_button.click(
        generate_image_diffusion_handler,
        inputs=[celebs_name_label, generate_image_prompt_text],
        outputs=[label_upload_here, celeb_generated_image]

    )

    describe_button.click(
        describe_handler,
        inputs=[input_key, org_id, mongo_prompt_read_config, cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, cloudinary_folder, mongo_config, mongo_connection_string, mongo_database, celebs_name_label, question_prompt, know_your_celeb_description, celeb_real_photo,  celeb_generated_image],
        outputs=[label_upload_here, celebs_name_label, question_prompt, know_your_celeb_description, celeb_real_photo, celeb_generated_image]
    )


    optimize_prompt_chatgpt_button.click(
        generate_optimized_prompt,
        inputs=[input_prompt],
        outputs=[input_prompt]
    )
    
   
   
    generate_button.click(
        create_image_from_prompt_handler,
        inputs=[input_key, org_id, input_prompt, input_imagesize, input_num_images],
        outputs=[label_picturize_it, output_generated_image, generated_images_gallery]
    )

    celeb_variation_button.click(
        create_variation_from_image_handler,
        inputs=[input_key, org_id, celeb_real_photo, input_imagesize, input_num_images],
        outputs=[label_upload_here, celeb_generated_image, generated_images_gallery]
    )
    
    generate_variations_button.click(
        create_variation_from_image_handler,
        inputs=[input_key, org_id, input_image_variation, input_imagesize, input_num_images],
        outputs=[label_get_variation, output_generated_image, generated_images_gallery]
    )

    generate_more_variations_button.click(
        create_variation_from_image_handler,
        inputs=[input_key, org_id, output_generated_image, input_imagesize, input_num_images],
        outputs=[label_upload_variation, output_generated_image, generated_images_gallery]
    )

    output_cloudinary_button.click(
        cloudinary_search,
        inputs=[cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, cloudinary_folder],
        outputs=[generated_images_gallery]
    )

    generated_images_gallery.select(
        generated_images_gallery_on_select,
        inputs=[generated_images_gallery],
        outputs=[output_generated_image]
    )

    variation_cloudinary_upload.click(
        cloudinary_upload,
        inputs=[cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, cloudinary_folder, output_generated_image, name_variation_it],
        outputs=[label_upload_variation,output_generated_image]
    )

    transcribe_button.click(
        transcribe_handler,
        inputs=[input_key, org_id, audio_file],
        outputs=[input_transcriptionprompt, input_prompt]
    )

    transcribe_whisper_large_v2_button.click(
        transcribe_whisper_large_v2,
        inputs=[audio_file],
        outputs=[input_transcriptionprompt, input_prompt]
    )

if __name__ == "__main__":
    AskMeTabbedScreen.launch()
