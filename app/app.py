import json
import os

import gradio as gr
from ExamplesUtil.CelebPromptGenerator import *
from MongoUtil.StateDataClient import *
from UIHandlers import AskMeUIHandlers
from Utils.Optimizers import Prompt_Optimizer
import gpt3_tokenizer

    
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

LABEL_GPT_CELEB_SCREEN = "Name, Describe, Preview and Upload"


def get_private_mongo_config():
    return os.getenv("P_MONGODB_URI"), os.getenv("P_MONGODB_DATABASE")

# Examples fn

def get_audio_examples():
    prompt_generator = CelebPromptGenerator()

    return prompt_generator.get_audio_examples()

def get_input_examples():
    prompt_generator = CelebPromptGenerator()
    return prompt_generator.get_input_examples()

def get_images_examples():

    prompt_generator = CelebPromptGenerator()
    return prompt_generator.get_images_examples()

def create_celeb_prompt(name_it):
    prompt_generator = CelebPromptGenerator()
    return prompt_generator.create_celeb_prompt(name_it)

def get_saved_prompts(keyword): 
    try:
        connection_string, database = get_private_mongo_config()           

        state_data_client = StateDataClient(connection_string, database)
        prompt, response = state_data_client.read_description_from_prompt(keyword)
        return prompt
    except Exception as err:
        return f"What is {keyword}, and how to use this in Python ? Give an example."

def get_awesome_chatgpt_prompts(awesome_chatgpt_prompt):
    prompt_generator = CelebPromptGenerator()
    awesome_chatgpt_act = prompt_generator.get_awesome_chatgpt_prompt(awesome_chatgpt_prompt)
    return awesome_chatgpt_act, awesome_chatgpt_prompt


def get_keyword_prompts():
    connection_string, database = get_private_mongo_config()           
    state_data_client = StateDataClient(connection_string, database)
    saved_prompts = state_data_client.list_saved_prompts("codex")
    return saved_prompts


prompt_generator = CelebPromptGenerator()
question_prompts = prompt_generator.get_questions()
input_examples = prompt_generator.get_input_examples()
audio_examples = prompt_generator.get_audio_examples()
images_examples = prompt_generator.get_images_examples()
celeb_names = prompt_generator.get_celebs()
saved_prompts = get_keyword_prompts() 
awesome_chatgpt_prompts = prompt_generator.get_all_awesome_chatgpt_prompts()

def generated_images_gallery_on_select(evt: gr.SelectData, generated_images_gallery):    
    if evt.index >= 0:
        name = generated_images_gallery[evt.index]["name"]
        output_generated_image =  name
        return output_generated_image
    else:
        image_utils = ImageUtils()
        return image_utils.fallback_image_implement()

def generate_optimized_prompt(plain_text):
    prompt_optimizer = Prompt_Optimizer()
    return prompt_optimizer.generate_optimized_prompt(plain_text);
	
def tokenizer_calc(prompt):
    if prompt:
        return f"{gpt3_tokenizer.count_tokens(prompt)},{len(prompt)}"

# UI Component handlers

def cloudinary_search(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, folder_name):
    uihandlers = AskMeUIHandlers()    
    uihandlers.set_cloudinary_config(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret)
    return uihandlers.cloudinary_search(folder_name)

    
def cloudinary_upload(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, folder_name, input_celeb_picture, celebrity_name):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_cloudinary_config(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret)
    return uihandlers.cloudinary_upload(folder_name, input_celeb_picture, celebrity_name)

    
def generate_image_stability_ai_handler(stability_api_key, name_it, question_prompt):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_stabilityai_config(stability_api_key)
    return uihandlers.generate_image_stability_ai_handler(name_it, question_prompt)

    
def generate_image_diffusion_handler(name_it):
    uihandlers = AskMeUIHandlers()
    return uihandlers.generate_image_diffusion_handler(name_it)
    
def transcribe_handler(api_key, org_id, audio_file):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_openai_config(api_key, org_id)
    return uihandlers.transcribe_handler(audio_file)
    
def create_variation_from_image_handler(api_key, org_id, input_image_variation, input_imagesize, input_num_images):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_openai_config(api_key, org_id)
    return uihandlers.create_variation_from_image_handler(input_image_variation, input_imagesize, input_num_images)

def awesome_prompts_handler(api_key, org_id, mongo_config, mongo_connection_string, mongo_database, prompt, keyword):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_openai_config(api_key, org_id, mongo_prompt_read_config)
    uihandlers.set_mongodb_config(mongo_config, mongo_connection_string, mongo_database)
    return uihandlers.ask_chatgpt(prompt, keyword,"awesome-prompts")

def ask_chatgpt_handler(api_key, org_id, mongo_config, mongo_connection_string, mongo_database, prompt, keyword):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_mongodb_config(mongo_config, mongo_connection_string, mongo_database)
    uihandlers.set_openai_config(api_key, org_id)
    return uihandlers.ask_chatgpt(prompt, keyword,"codex")

def describe_handler(api_key, org_id, cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, cloudinary_folder, mongo_config, mongo_connection_string, mongo_database, name_it, question_prompt, input_celeb_real_picture, input_celeb_generated_picture):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_openai_config(api_key, org_id)
    uihandlers.set_mongodb_config(mongo_config, mongo_connection_string, mongo_database)
    uihandlers.set_cloudinary_config(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret)
    return uihandlers.describe_handler(name_it, question_prompt, cloudinary_folder, input_celeb_real_picture, input_celeb_generated_picture)

def get_celebs_response_handler(mongo_config, mongo_connection_string, mongo_database, keyword):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_mongodb_config(mongo_config, mongo_connection_string, mongo_database)
    return uihandlers.get_celebs_response_handler(keyword)

def create_image_from_prompt_handler(api_key, org_id, input_prompt, input_imagesize, input_num_images):
    uihandlers = AskMeUIHandlers()
    uihandlers.set_openai_config(api_key, org_id)
    return uihandlers.create_image_from_prompt_handler(input_prompt, input_imagesize, input_num_images)

with gr.Blocks(css='styles.css') as AskMeTabbedScreen:
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
        with gr.Row():
                input_num_images = gr.Slider(minimum=1,maximum=10,step=1,
                    label="Number of Images to generate", value=1, info="OpenAI API supports 1-10 images")
                input_imagesize = gr.Dropdown(["1024x1024", "512x512", "256x256"], value="256x256", label="Image size",
                                              info="Select one, use download for image size from Image generation/variation Output tab")
    with gr.Tab("Record, transcribe, picturize and upload"):
        gr.HTML("<p>Record voice, transcribe a prompt, picturize the prompt, create variations, get description of a celebrity and upload</p>")
        with gr.Tab("Whisper(whisper-1)"):
            with gr.Row():
                with gr.Column(scale=3):
                    audio_file = gr.Audio(
                        label="Record to describe what you want to picturize? and click on Transcribe",
                        source="microphone",
                        type="filepath"
                    )
                    audio_file = gr.Audio(
                        label="Upload Audio and Transcribe",
                        type="filepath"
                    )
                with gr.Column(scale=2):
                    gr.Examples(
                        examples=audio_examples,                   
                        label="Select one from Audio Examples and Transcribe",
                        fn=get_audio_examples,
                        examples_per_page=4,
                        inputs=audio_file)
                    transcribe_button = gr.Button("Transcribe via Whisper")  
            input_transcriptionprompt = gr.Label(label="Transcription Text")
        with gr.Tab("Image generation"):
            input_prompt = gr.Textbox(label="Prompt Text to describe what you want to picturize?", lines=7)
            with gr.Row():                
                with gr.Column(scale=1):                    
                    input_prompt_label = gr.Label(label="Token,Characters",value="Tokenizer")
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
        with gr.Tab("GPT(gpt-3.5-turbo)"):
            with gr.Row():
                with gr.Column(scale=4):  
                    name_it = gr.Textbox(label="Name")                   
                    question_prompt = gr.Textbox(label="Prompt", lines=2)
                    with gr.Accordion("Celebrity Examples, select one from here", open=False):

                        gr.Examples(
                            fn=create_celeb_prompt,
                            label="Select one from a celebrity",
                            examples=celeb_names,
                            examples_per_page=50,
                            inputs=[name_it],
                            outputs=[question_prompt],                
                            cache_examples=True,
                        )
                with gr.Column(scale=1):
                    identify_celeb_button = gr.Button("Get Details, if exists")
                    with gr.Accordion("Generate image", open=False):
                        generate_image_stability_ai_button = gr.Button("via Stability AI")
                        generate_image_diffusion_button = gr.Button("*via stable-diffusion-2 model")
                        label_generate_image_diffusion = gr.Label(value="* takes 40-50 mins on CPU", label="Warning") 

                        celeb_variation_button = gr.Button("variation from the real photo (DALL-E 2)")
                    describe_button = gr.Button("Describe via ChatGPT and Save")
                    label_describe_gpt = gr.Label(value="Generate or Upload Image to Save", label="Info")
            with gr.Row():
                celeb_real_photo = gr.Image(label="Real Photo",  type="filepath")                        
                celeb_generated_image = gr.Image(label="Generated Image",  type="filepath")
            with gr.Row():                
                know_your_celeb_description = gr.Textbox(label="Description", lines=7)                

            label_upload_here = gr.Label(value=LABEL_GPT_CELEB_SCREEN, label="Info")
        with gr.Tab("Output"):
            with gr.Row():            
                with gr.Column(scale=4):
                    with gr.Accordion("Generated Gallery", open=False):
                        generated_images_gallery = gr.Gallery(
                                        label="Generated Images").style(preview="False", columns=[4])
                    output_generated_image = gr.Image(label="Preview Image",  type="filepath")
                with gr.Column(scale=1):
                    output_celebs_button = gr.Button("Get celebrity images from Cloudinary")
                    generate_more_variations_button = gr.Button("More variations via DALL-E")
                    name_variation_it = gr.Textbox(label="Name variation to upload")   
                    variation_cloudinary_upload = gr.Button("Upload to Cloudinary")
            label_upload_variation = gr.Label(value="Upload output", label="Output Info")
    with gr.Tab("Ask Codex"):
        with gr.Row():
            with gr.Column(scale=4):
                ask_keyword = gr.Textbox(label="Keyword", lines=1)
                ask_prompt = gr.Textbox(label="Prompt", lines=5)                
            with gr.Column(scale=1):
                gr.Examples(
                    label="Recent keywords",
                    fn=get_saved_prompts,
                    examples=saved_prompts,
                    examples_per_page=10,
                    inputs=[ask_keyword],
                    outputs=[ask_prompt],   
                    cache_examples=True,
                )                
                chat_gpt_prompt_label = gr.Label(label="Token,Characters",value="Tokenizer")
                ask_chatgpt_button = gr.Button("Ask ChatGPT")                
        with gr.Row():
            keyword_response_code = gr.Code(label="Code", language="python", lines=7)
        label_codex_here = gr.Label(value="Ask Codex, Write a better code", label="Info")
    with gr.Tab("🧠 Awesome ChatGPT Prompts"):
        gr.HTML(AWESOME_CHATGPT_PROMPTS)
        with gr.Row():
            with gr.Column(scale=4):
                awesome_chatgpt_act = gr.Textbox(label="Act")
                awesome_chatgpt_prompt = gr.Textbox(label="Awesome ChatGPT Prompt", lines=5)                  
            with gr.Column(scale=1):
                label_awesome_chatgpt_here = gr.Label(value="See examples below", label="Info")
                awesome_chatgpt_prompt_label = gr.Label(label="Token,Characters",value="Tokenizer")
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
                    inputs=[awesome_chatgpt_prompt],
                    outputs=[awesome_chatgpt_act, awesome_chatgpt_prompt],   
                    cache_examples=True,
                )
    with gr.Tab("DISCLAIMER"):
        gr.Markdown(DISCLAIMER)
    gr.HTML(FOOTER)

    awesome_chatgpt_prompt.change(
        fn=tokenizer_calc,
        inputs=[awesome_chatgpt_prompt],
        outputs=[awesome_chatgpt_prompt_label]        
    )
    
    
    input_prompt.change(
        fn=tokenizer_calc,
        inputs=[input_prompt],
        outputs=[input_prompt_label]        
    )
    
    ask_prompt.change(
        fn=tokenizer_calc,
        inputs=[ask_prompt],
        outputs=[chat_gpt_prompt_label]        
    )
    
    generate_image_stability_ai_button.click(
        generate_image_stability_ai_handler,
        inputs=[stability_api_key, name_it, question_prompt],
        outputs=[label_upload_here,celeb_generated_image]
    )
    
    generate_image_diffusion_button.click(
        generate_image_diffusion_handler,
        inputs=[name_it],
        outputs=[label_upload_here, celeb_generated_image]
    )

    describe_button.click(
        describe_handler,
        inputs=[input_key, org_id, cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, cloudinary_folder, mongo_config, mongo_connection_string, mongo_database, name_it, question_prompt, celeb_real_photo,  celeb_generated_image],
        outputs=[label_upload_here, name_it, question_prompt, know_your_celeb_description, celeb_real_photo, celeb_generated_image]
    )

    optimize_prompt_chatgpt_button.click(
        generate_optimized_prompt,
        inputs=[input_prompt],
        outputs=[input_prompt]
    )
    
    ask_awesome_chatgpt_button.click(
        awesome_prompts_handler,
        inputs=[input_key, org_id, mongo_config, mongo_connection_string, mongo_database, awesome_chatgpt_prompt, awesome_chatgpt_act],
        outputs=[label_awesome_chatgpt_here, awesome_chatgpt_response]
    )

    ask_chatgpt_button.click(
        ask_chatgpt_handler,
        inputs=[input_key, org_id, mongo_config, mongo_connection_string, mongo_database, ask_prompt, ask_keyword],
        outputs=[label_codex_here, keyword_response_code]
    )    
    
    identify_celeb_button.click(
        get_celebs_response_handler,
        inputs=[mongo_config, mongo_connection_string, mongo_database, name_it],
        outputs=[name_it, question_prompt, know_your_celeb_description, celeb_real_photo, celeb_generated_image]
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

    output_celebs_button.click(
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

if __name__ == "__main__":
    AskMeTabbedScreen.launch()