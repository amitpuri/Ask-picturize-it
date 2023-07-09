import os
import csv
from os.path import isfile, join


class PromptGenerator:
    
    def __init__(self):
        self.examples_file = 'prompt-texts/examples.csv'
        self.prompts_file = 'prompt-texts/prompts.csv'
        self.images_path = 'images'

    def get_input_examples(self):
        with open(self.examples_file, newline='') as examples_file:
            examples = csv.DictReader(examples_file, delimiter=',')
            input_examples=[]
            for example in examples: 
                input_examples.append([example['Prompt']])
        return input_examples

    def get_audio_examples(self, lang = 'english'):
        self.audio_path = join('audio', lang)
        if os.path.exists(self.audio_path):
            audio_examples = [join(self.audio_path, f) for f in os.listdir(self.audio_path) if isfile(join(self.audio_path, f))]
            return audio_examples
        else:
            return []

    def get_images_examples(self):
        if os.path.exists(self.images_path):
            images_examples = [join(self.images_path, f) for f in os.listdir(self.images_path) if isfile(join(self.images_path, f))]
            return images_examples
        else:
            return []
    
    def get_awesome_chatgpt_prompt(self, awesome_chatgpt_act):
        with open(self.prompts_file, encoding='unicode_escape') as prompts_file:
            prompts = csv.DictReader(prompts_file, delimiter=',')            
            for prompt in prompts: 
                if prompt['act'] == awesome_chatgpt_act:
                    return prompt['prompt']
        

    def get_all_awesome_chatgpt_prompts(self, prompttype="fav"):
        with open(self.prompts_file, encoding='unicode_escape') as prompts_file:
            prompts = csv.DictReader(prompts_file, delimiter=',')
            awesome_chatgpt_prompts = []
            for prompt in prompts: 
                if prompt['prompttype']==prompttype:
                    awesome_chatgpt_prompts.append([prompt['act']])
        return awesome_chatgpt_prompts