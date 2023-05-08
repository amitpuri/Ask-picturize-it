import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Prompt_Optimizer: 
    def __init__(self):
        self.prompter_model, self.prompter_tokenizer = self.load_prompter()

    def load_prompter(self):
        prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return prompter_model, tokenizer


    def generate_optimized_prompt(self, plain_text: str):
        input_ids = self.prompter_tokenizer(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids
        eos_id = self.prompter_tokenizer.eos_token_id
        outputs = self.prompter_model.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
        output_texts = self.prompter_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
        return res
