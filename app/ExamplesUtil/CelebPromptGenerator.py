import os
import csv
from os.path import isfile, join


class CelebPromptGenerator:
    
    def __init__(self):
        self.questions_file = 'prompt-texts/questions.csv'
        self.examples_file = 'prompt-texts/examples.csv'
        self.celebrities_file = 'prompt-texts/celebrities.csv'
        self.prompts_file = 'prompt-texts/prompts.csv'
        self.audio_path = 'audio'
        self.images_path = 'images'
        
    def get_question_prompts(self):
        question_prompts = self.get_questions()
        return question_prompts[0]

    def create_celeb_prompt(self, name_it):
        question_prompts = self.get_question_prompts()
        return f"{question_prompts['Prompt']} {question_prompts['Conjunction']} {name_it}"

    def parse_celeb_prompt(self, input_prompt, name_it):
        return f"{input_prompt['Prompt']} {input_prompt['Conjunction']} {name_it}"

    def get_questions(self):
        with open(self.questions_file, newline='') as questions_file:
            questions = csv.DictReader(questions_file, delimiter=',')
            question_prompts = []
            for question in questions: 
                question_item = {'Prompt': f"{question['Prompt']}", 'Conjunction': f"{question['Conjunction']}"}
                question_prompts.append(question_item)
        return question_prompts
        
    def read_questions(self):
        with open(self.questions_file, newline='') as questions_file:
            questions = csv.DictReader(questions_file, delimiter=',')
            question_prompts = []
            for question in questions: 
                question_prompts.append([question['Prompt']])
        return question_prompts

    def get_input_examples(self):
        with open(self.examples_file, newline='') as examples_file:
            examples = csv.DictReader(examples_file, delimiter=',')
            input_examples=[]
            for example in examples: 
                input_examples.append([example['Prompt']])
        return input_examples

    def get_audio_examples(self):
        audio_examples = [join(self.audio_path, f) for f in os.listdir(self.audio_path) if isfile(join(self.audio_path, f))]
        return audio_examples

    def get_images_examples(self):
        images_examples = [join(self.images_path, f) for f in os.listdir(self.images_path) if isfile(join(self.images_path, f))]
        return images_examples
    
    def get_celebs(self):
        celebs = []
        with open(self.celebrities_file, newline='') as celebrities_file:
            celeb_names = csv.DictReader(celebrities_file, delimiter=',')
            for celeb in celeb_names:
                celebs.append([celeb['Celebrity Name']])
        return celebs

    def get_awesome_chatgpt_prompt(self, awesome_chatgpt_prompt):
        with open(self.prompts_file, newline='') as prompts_file:
            prompts = csv.DictReader(prompts_file, delimiter=',')            
            for prompt in prompts: 
                if prompt['prompt'] == awesome_chatgpt_prompt:
                    return prompt['act']
        

    def get_all_awesome_chatgpt_prompts(self):
        with open(self.prompts_file, newline='') as prompts_file:
            prompts = csv.DictReader(prompts_file, delimiter=',')
            awesome_chatgpt_prompts = []
            for prompt in prompts: 
                awesome_chatgpt_prompts.append([prompt['prompt']])
        return awesome_chatgpt_prompts