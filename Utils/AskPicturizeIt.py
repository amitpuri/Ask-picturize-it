import wikipedia
import requests
import json

class AskPicturizeIt:
    TITLE = '# [Ask-me-to-picturize-it](https://github.com/amitpuri/Ask-me-to-picturize-it)'
    DESCRIPTION = """<strong>This space uses following:</strong>
       <p>
       <ul>
    
       <li>OpenAI API Whisper(whisper-1) <a href='https://openai.com/research/whisper'>https://openai.com/research/whisper</a></li>
    
       <li>DALL-E <a href='https://openai.com/product/dall-e-2'>https://openai.com/product/dall-e-2</a></li>
       <li>GPT(gpt-3.5-turbo) <a href='https://openai.com/product/gpt-4'>https://openai.com/product/gpt-4</a></li>
    
       <li>Cloudinary <a href='https://cloudinary.com/documentation/python_quickstart'>https://cloudinary.com/documentation/python_quickstart</a></li>
       <li>Gradio App <a href='https://gradio.app/docs'>https://gradio.app/docs</a> in Python and MongoDB</li>
       <li>Prompt optimizer <a href='https://huggingface.co/microsoft/Promptist'>https://huggingface.co/microsoft/Promptist</a></li>
       <li>stabilityai/stable-diffusion-2-1 <a href='https://huggingface.co/stabilityai/stable-diffusion-2-1'>https://huggingface.co/stabilityai/stable-diffusion-2-1</a></li>
       <li>Stability AI <a href='https://stability.ai'>https://stability.ai</a></li>
       <li>LangChain OpenAI <a href='https://js.langchain.com/docs/getting-started/guide-llm'>https://js.langchain.com/docs/getting-started/guide-llm</a></li>
       <li>Article Extractor and Summarizer on Rapid API <a href='https://rapidapi.com'>https://rapidapi.com</a></li>   
       
       
       </ul>
       </p>
     """
    RESEARCH_SECTION = """
       <p><strong>Check it out</strong>
    
       </p>

       <p>
       <ul>
       <li><p>Attention Is All You Need <a href='https://arxiv.org/abs/1706.03762'>https://arxiv.org/abs/1706.03762</a></p></li>
       <li><p>NLP's ImageNet moment has arrived <a href='https://thegradient.pub/nlp-imagenet'>https://thegradient.pub/nlp-imagenet</a></p></li>   
       <li><p>Zero-Shot Text-to-Image Generation <a href='https://arxiv.org/abs/2102.12092'>https://arxiv.org/abs/2102.12092</a></p></li>   
       <li><p>Transformer: A Novel Neural Network Architecture for Language Understanding <a href='https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html'>https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html</a></p></li>
       <li><p>CS25: Transformers United V2 <a href='https://web.stanford.edu/class/cs25'>https://web.stanford.edu/class/cs25</a></p></li>
       <li><p>CS25: Stanford Seminar - Transformers United 2023: Introduction to Transformer <a href='https://youtu.be/XfpMkf4rD6E'>https://youtu.be/XfpMkf4rD6E</a></p></li>
       <li><p>Temperature in NLP <a href='https://lukesalamone.github.io/posts/what-is-temperature'>https://lukesalamone.github.io/posts/what-is-temperature</a></p></li>
       <li><p>LangChain <a href='https://langchain.com/features.html'>https://langchain.com/features.html</a></p></li>
       <li><p>LangChain Python <a href='https://python.langchain.com'>https://python.langchain.com</a></p></li>
       <li><p>An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <a href='https://arxiv.org/abs/2010.11929'>https://arxiv.org/abs/2010.11929</a></p></li>
       <li>stable-diffusion-image-variations <a href='https://huggingface.co/spaces/lambdalabs/stable-diffusion-image-variations'>https://huggingface.co/spaces/lambdalabs/stable-diffusion-image-variations</a></li> 
       <li>text-to-avatar <a href='https://huggingface.co/spaces/lambdalabs/text-to-avatar'>https://huggingface.co/spaces/lambdalabs/text-to-avatar</a></li> 
       <li>generative-music-visualizer <a href='https://huggingface.co/spaces/lambdalabs/generative-music-visualizer'>https://huggingface.co/spaces/lambdalabs/generative-music-visualizer</a></li> 
       <li>text-to-pokemon <a href='https://huggingface.co/spaces/lambdalabs/text-to-pokemon'>https://huggingface.co/spaces/lambdalabs/text-to-pokemon</a></li> 
       <li>image-mixer-demo <a href='https://huggingface.co/spaces/lambdalabs/image-mixer-demo'>https://huggingface.co/spaces/lambdalabs/image-mixer-demo</a></li> 
       <li>Stable Diffusion <a href='https://huggingface.co/blog/stable_diffusion'>https://huggingface.co/blog/stable_diffusion</a></li> 
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


   
    def get_wikimedia_image(self, keyword):
        WIKI_REQUEST = 'http://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles='

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
                    try:
                        wkpage = wikipedia.WikipediaPage(title = result[0])
                    except:
                        print(result)
                    finally:
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
    
    def get_wiki_page_summary(self, keyword):
        if keyword:
            try:
                return wikipedia.page(keyword).summary
            except wikipedia.exceptions.PageError:
                return f"No page for this keyword {keyword}"
            except Exception as exception:
                print(f"Exception Name: {type(exception).__name__}")
                print(exception)